/**
 * Copyright (c) 2016 Davinder Singh (DSM_) <ds.mudhar<@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "motion_estimation.h"
#include "libavcodec/mathops.h"
#include "libavutil/common.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/motion_vector.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

typedef struct MEContext {
    const AVClass *class;
    AVMotionEstContext me_ctx;
    int method;                         ///< motion estimation method

    int mb_size;                        ///< macroblock size
    int search_param;                   ///< search parameter
    int onlyprev;                       ///< onlyprev parameter
    int b_width, b_height, b_count;
    int log2_mb_size;

    AVFrame *prev, *cur, *next;
    int current_dir;

    int (*mv_table[3])[2][2];           ///< motion vectors of current & prev 2 frames

    int vr;                             // New flag for VR mode
} MEContext;

typedef struct ThreadData {
    AVMotionEstContext *me_ctx;
    AVMotionVector *mv_data;
} ThreadData;

#define OFFSET(x) offsetof(MEContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CONST(name, help, val, u) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, 0, 0, FLAGS, .unit = u }

static const AVOption mestimate_options[] = {
    { "method", "motion estimation method", OFFSET(method), AV_OPT_TYPE_INT, {.i64 = AV_ME_METHOD_ESA}, AV_ME_METHOD_ESA, AV_ME_METHOD_UMH, FLAGS, .unit = "method" },
        CONST("esa",   "exhaustive search",                  AV_ME_METHOD_ESA,      "method"),
        CONST("tss",   "three step search",                  AV_ME_METHOD_TSS,      "method"),
        CONST("tdls",  "two dimensional logarithmic search", AV_ME_METHOD_TDLS,     "method"),
        CONST("ntss",  "new three step search",              AV_ME_METHOD_NTSS,     "method"),
        CONST("fss",   "four step search",                   AV_ME_METHOD_FSS,      "method"),
        CONST("ds",    "diamond search",                     AV_ME_METHOD_DS,       "method"),
        CONST("hexbs", "hexagon-based search",               AV_ME_METHOD_HEXBS,    "method"),
        CONST("epzs",  "enhanced predictive zonal search",   AV_ME_METHOD_EPZS,     "method"),
        CONST("umh",   "uneven multi-hexagon search",        AV_ME_METHOD_UMH,      "method"),
    { "mb_size", "macroblock size", OFFSET(mb_size), AV_OPT_TYPE_INT, {.i64 = 16}, 8, INT_MAX, FLAGS },
    { "search_param", "search parameter", OFFSET(search_param), AV_OPT_TYPE_INT, {.i64 = 7}, 4, INT_MAX, FLAGS },
    { "onlyprev", "use only previous frame for motion estimation", OFFSET(onlyprev), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, FLAGS },
    { "vr", "enable VR left-right eye comparison", OFFSET(vr), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(mestimate);

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
    AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_YUVJ411P,
    AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUVA444P,
    AV_PIX_FMT_GRAY8,
    AV_PIX_FMT_NONE
};

static int config_input(AVFilterLink *inlink)
{
    MEContext *s = inlink->dst->priv;
    int i;

    s->log2_mb_size = av_ceil_log2_c(s->mb_size);
    s->mb_size = 1 << s->log2_mb_size;

    s->b_width  = inlink->w >> s->log2_mb_size;
    s->b_height = inlink->h >> s->log2_mb_size;
    s->b_count = s->b_width * s->b_height;

    if (s->b_count == 0)
        return AVERROR(EINVAL);

    for (i = 0; i < 3; i++) {
        s->mv_table[i] = av_calloc(s->b_count, sizeof(*s->mv_table[0]));
        if (!s->mv_table[i])
            return AVERROR(ENOMEM);
    }

    ff_me_init_context(&s->me_ctx, s->mb_size, s->search_param, inlink->w, inlink->h, 0, (s->b_width - 1) << s->log2_mb_size, 0, (s->b_height - 1) << s->log2_mb_size);

    return 0;
}

static void add_mv_data(AVMotionVector *mv, int mb_size,
                        int x, int y, int x_mv, int y_mv, int dir)
{
    mv->w = mb_size;
    mv->h = mb_size;
    mv->dst_x = x + (mb_size >> 1);
    mv->dst_y = y + (mb_size >> 1);
    mv->src_x = x_mv + (mb_size >> 1);
    mv->src_y = y_mv + (mb_size >> 1);
    mv->source = dir ? 1 : -1;
    mv->flags = 0;
}

#define SEARCH_MV_SLICE(method)\
    do {\
        for (mb_y = slice_start; mb_y < slice_end; mb_y++)\
            for (mb_x = 0; mb_x < s->b_width; mb_x++) {\
                const int x_mb = (mb_x << s->log2_mb_size);\
                const int y_mb = (mb_y << s->log2_mb_size);\
                int mv[2] = {x_mb, y_mb};\
                ff_me_search_##method(me_ctx, x_mb, y_mb, mv);\
                add_mv_data(td->mv_data + mv_count++, s->mb_size, x_mb, y_mb, mv[0], mv[1], dir);\
            }\
    } while (0)

#define ADD_PRED(preds, px, py)\
    do {\
        preds.mvs[preds.nb][0] = px;\
        preds.mvs[preds.nb][1] = py;\
        preds.nb++;\
    } while(0)


static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    MEContext *s = ctx->priv;
    ThreadData *td = arg;
    AVMotionEstContext *me_ctx = td->me_ctx;
    int mb_y, mb_x;
    int dir = s->current_dir;

    int slice_size = s->b_height / nb_jobs;
    if (slice_size * nb_jobs < s->b_height)
        slice_size++;
    int slice_start = jobnr * slice_size;
    int slice_end = min(slice_start + slice_size, s->b_height);
    int mv_count = slice_start * s->b_width + (dir * s->b_height * s->b_width);

    if (s->method == AV_ME_METHOD_DS)
        SEARCH_MV_SLICE(ds);
    else if (s->method == AV_ME_METHOD_ESA)
        SEARCH_MV_SLICE(esa);
    else if (s->method == AV_ME_METHOD_FSS)
        SEARCH_MV_SLICE(fss);
    else if (s->method == AV_ME_METHOD_NTSS)
        SEARCH_MV_SLICE(ntss);
    else if (s->method == AV_ME_METHOD_TDLS)
        SEARCH_MV_SLICE(tdls);
    else if (s->method == AV_ME_METHOD_TSS)
        SEARCH_MV_SLICE(tss);
    else if (s->method == AV_ME_METHOD_HEXBS)
        SEARCH_MV_SLICE(hexbs);
    else if (s->method == AV_ME_METHOD_UMH) {
        for (mb_y = slice_start; mb_y < slice_end; mb_y++)
            for (mb_x = 0; mb_x < s->b_width; mb_x++) {
                const int mb_i = mb_x + mb_y * s->b_width;
                const int x_mb = mb_x << s->log2_mb_size;
                const int y_mb = mb_y << s->log2_mb_size;
                int mv[2] = {x_mb, y_mb};

                AVMotionEstPredictor *preds = me_ctx->preds;
                preds[0].nb = 0;

                ADD_PRED(preds[0], 0, 0);

                //left mb in current frame
                if (mb_x > 0)
                    ADD_PRED(preds[0], s->mv_table[0][mb_i - 1][dir][0], s->mv_table[0][mb_i - 1][dir][1]);

                if (mb_y > 0) {
                    //top mb in current frame
                    ADD_PRED(preds[0], s->mv_table[0][mb_i - s->b_width][dir][0], s->mv_table[0][mb_i - s->b_width][dir][1]);

                    //top-right mb in current frame
                    if (mb_x + 1 < s->b_width)
                        ADD_PRED(preds[0], s->mv_table[0][mb_i - s->b_width + 1][dir][0], s->mv_table[0][mb_i - s->b_width + 1][dir][1]);
                    //top-left mb in current frame
                    else if (mb_x > 0)
                        ADD_PRED(preds[0], s->mv_table[0][mb_i - s->b_width - 1][dir][0], s->mv_table[0][mb_i - s->b_width - 1][dir][1]);
                }

                //median predictor
                if (preds[0].nb == 4) {
                    me_ctx->pred_x = mid_pred(preds[0].mvs[1][0], preds[0].mvs[2][0], preds[0].mvs[3][0]);
                    me_ctx->pred_y = mid_pred(preds[0].mvs[1][1], preds[0].mvs[2][1], preds[0].mvs[3][1]);
                } else if (preds[0].nb == 3) {
                    me_ctx->pred_x = mid_pred(0, preds[0].mvs[1][0], preds[0].mvs[2][0]);
                    me_ctx->pred_y = mid_pred(0, preds[0].mvs[1][1], preds[0].mvs[2][1]);
                } else if (preds[0].nb == 2) {
                    me_ctx->pred_x = preds[0].mvs[1][0];
                    me_ctx->pred_y = preds[0].mvs[1][1];
                } else {
                    me_ctx->pred_x = 0;
                    me_ctx->pred_y = 0;
                }

                ff_me_search_umh(me_ctx, x_mb, y_mb, mv);

                s->mv_table[0][mb_i][dir][0] = mv[0] - x_mb;
                s->mv_table[0][mb_i][dir][1] = mv[1] - y_mb;
                add_mv_data(td->mv_data + mv_count++, s->mb_size, x_mb, y_mb, mv[0], mv[1], dir);
            }

    } else if (s->method == AV_ME_METHOD_EPZS) {

        for (mb_y = slice_start; mb_y < slice_end; mb_y++)
            for (mb_x = 0; mb_x < s->b_width; mb_x++) {
                const int mb_i = mb_x + mb_y * s->b_width;
                const int x_mb = mb_x << s->log2_mb_size;
                const int y_mb = mb_y << s->log2_mb_size;
                int mv[2] = {x_mb, y_mb};

                AVMotionEstPredictor *preds = me_ctx->preds;
                preds[0].nb = 0;
                preds[1].nb = 0;

                ADD_PRED(preds[0], 0, 0);

                //left mb in current frame
                if (mb_x > 0)
                    ADD_PRED(preds[0], s->mv_table[0][mb_i - 1][dir][0], s->mv_table[0][mb_i - 1][dir][1]);

                //top mb in current frame
                if (mb_y > 0)
                    ADD_PRED(preds[0], s->mv_table[0][mb_i - s->b_width][dir][0], s->mv_table[0][mb_i - s->b_width][dir][1]);

                //top-right mb in current frame
                if (mb_y > 0 && mb_x + 1 < s->b_width)
                    ADD_PRED(preds[0], s->mv_table[0][mb_i - s->b_width + 1][dir][0], s->mv_table[0][mb_i - s->b_width + 1][dir][1]);

                //median predictor
                if (preds[0].nb == 4) {
                    me_ctx->pred_x = mid_pred(preds[0].mvs[1][0], preds[0].mvs[2][0], preds[0].mvs[3][0]);
                    me_ctx->pred_y = mid_pred(preds[0].mvs[1][1], preds[0].mvs[2][1], preds[0].mvs[3][1]);
                } else if (preds[0].nb == 3) {
                    me_ctx->pred_x = mid_pred(0, preds[0].mvs[1][0], preds[0].mvs[2][0]);
                    me_ctx->pred_y = mid_pred(0, preds[0].mvs[1][1], preds[0].mvs[2][1]);
                } else if (preds[0].nb == 2) {
                    me_ctx->pred_x = preds[0].mvs[1][0];
                    me_ctx->pred_y = preds[0].mvs[1][1];
                } else {
                    me_ctx->pred_x = 0;
                    me_ctx->pred_y = 0;
                }

                //collocated mb in prev frame
                ADD_PRED(preds[0], s->mv_table[1][mb_i][dir][0], s->mv_table[1][mb_i][dir][1]);

                //accelerator motion vector of collocated block in prev frame
                ADD_PRED(preds[1], s->mv_table[1][mb_i][dir][0] + (s->mv_table[1][mb_i][dir][0] - s->mv_table[2][mb_i][dir][0]),
                                    s->mv_table[1][mb_i][dir][1] + (s->mv_table[1][mb_i][dir][1] - s->mv_table[2][mb_i][dir][1]));

                //left mb in prev frame
                if (mb_x > 0)
                    ADD_PRED(preds[1], s->mv_table[1][mb_i - 1][dir][0], s->mv_table[1][mb_i - 1][dir][1]);

                //top mb in prev frame
                if (mb_y > 0)
                    ADD_PRED(preds[1], s->mv_table[1][mb_i - s->b_width][dir][0], s->mv_table[1][mb_i - s->b_width][dir][1]);

                //right mb in prev frame
                if (mb_x + 1 < s->b_width)
                    ADD_PRED(preds[1], s->mv_table[1][mb_i + 1][dir][0], s->mv_table[1][mb_i + 1][dir][1]);

                //bottom mb in prev frame
                if (mb_y + 1 < s->b_height)
                    ADD_PRED(preds[1], s->mv_table[1][mb_i + s->b_width][dir][0], s->mv_table[1][mb_i + s->b_width][dir][1]);

                ff_me_search_epzs(me_ctx, x_mb, y_mb, mv);

                s->mv_table[0][mb_i][dir][0] = mv[0] - x_mb;
                s->mv_table[0][mb_i][dir][1] = mv[1] - y_mb;
                add_mv_data(td->mv_data + mv_count++, s->mb_size, x_mb, y_mb, mv[0], mv[1], dir);
            }
    }

    return 0;
}

static void prepare_reference_frame(AVFrame *prev_frame, AVFrame *cur_frame)
{
    if (!prev_frame || !cur_frame)
        return;

    const int width = cur_frame->width;
    const int height = cur_frame->height;
    const int half_width = width / 2;

    // Process line by line
    for (int y = 0; y < height; y++) {
        // Left side stays as previous frame (unchanged)
        // Right side: copy from current frame's left side
        memcpy(prev_frame->data[0] + y * prev_frame->linesize[0] + half_width,
               cur_frame->data[0] + y * cur_frame->linesize[0],
               half_width);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    MEContext *s = ctx->priv;
    AVMotionEstContext *me_ctx = &s->me_ctx;
    AVFrameSideData *sd;
    AVFrame *out;
    int ret;

    if (frame->pts == AV_NOPTS_VALUE) {
        ret = ff_filter_frame(ctx->outputs[0], frame);
        return ret;
    }

    av_frame_free(&s->prev);
    s->prev = s->cur;
    s->cur  = s->next;
    s->next = frame;

    s->mv_table[2] = memcpy(s->mv_table[2], s->mv_table[1], sizeof(*s->mv_table[1]) * s->b_count);
    s->mv_table[1] = memcpy(s->mv_table[1], s->mv_table[0], sizeof(*s->mv_table[0]) * s->b_count);

    if (s->vr) {
        prepare_reference_frame(s->prev, s->cur);
    }

    if (!s->cur) {
        s->cur = av_frame_clone(frame);
        if (!s->cur)
            return AVERROR(ENOMEM);
    }

    if (!s->prev)
        return 0;

    out = av_frame_clone(s->cur);
    if (!out)
        return AVERROR(ENOMEM);

    sd = av_frame_new_side_data(out, AV_FRAME_DATA_MOTION_VECTORS, (s->onlyprev ? 1 : 2) * s->b_count * sizeof(AVMotionVector));
    if (!sd) {
        av_frame_free(&out);
        return AVERROR(ENOMEM);
    }

    me_ctx->data_cur = s->cur->data[0];
    me_ctx->linesize = s->cur->linesize[0];

    ThreadData td;
    td.me_ctx = &s->me_ctx;
    td.mv_data = (AVMotionVector *)sd->data;

    s->current_dir = 0;
    me_ctx->data_ref = s->prev->data[0];
    ff_filter_execute(ctx, filter_slice, &td, NULL, FFMIN(s->b_height, ff_filter_get_nb_threads(ctx)));
    if (!s->onlyprev)
    {
        s->current_dir = 1;
        me_ctx->data_ref = s->next->data[0];
        ff_filter_execute(ctx, filter_slice, &td, NULL, FFMIN(s->b_height, ff_filter_get_nb_threads(ctx)));
    }

    return ff_filter_frame(ctx->outputs[0], out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MEContext *s = ctx->priv;
    int i;

    av_frame_free(&s->prev);
    av_frame_free(&s->cur);
    av_frame_free(&s->next);

    for (i = 0; i < 3; i++)
        av_freep(&s->mv_table[i]);
}

static const AVFilterPad mestimate_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .filter_frame  = filter_frame,
        .config_props  = config_input,
    },
};

const AVFilter ff_vf_mestimate = {
    .name          = "mestimate",
    .description   = NULL_IF_CONFIG_SMALL("Generate motion vectors."),
    .priv_size     = sizeof(MEContext),
    .priv_class    = &mestimate_class,
    .uninit        = uninit,
    .flags         = AVFILTER_FLAG_METADATA_ONLY | AVFILTER_FLAG_SLICE_THREADS,
    FILTER_INPUTS(mestimate_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
