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

#include "motion_estimation2.h"
#include "libavcodec/mathops.h"
#include "libavutil/common.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/motion_vector.h"
#include "avfilter.h"
#include "internal.h"
#include "video.h"

typedef struct MEContext {
    const AVClass *class;
    AVMotionEstContext2 me_ctx;

    int mb_size;                        ///< macroblock size
    int search_param;                   ///< search parameter
    int b_width, b_height, b_count;
    int log2_mb_size;

    AVFrame *prev, *cur, *next;
    int current_dir;

    int (*mv_table[3])[2][2];           ///< motion vectors of current & prev 2 frames
} MEContext;

typedef struct ThreadData {
    AVMotionEstContext2 *me_ctx;
    AVMotionVector *mv_data;
} ThreadData;

#define OFFSET(x) offsetof(MEContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CONST(name, help, val, u) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, 0, 0, FLAGS, .unit = u }

static const AVOption mestimate2_options[] = {
    { "mb_size", "macroblock size", OFFSET(mb_size), AV_OPT_TYPE_INT, {.i64=16}, 8, INT_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(mestimate2);

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


static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    MEContext *s = ctx->priv;
    ThreadData *td = arg;
    AVMotionEstContext2 *me_ctx = td->me_ctx;
    int mb_y, mb_x;
    int dir = s->current_dir;

    int slice_size = s->b_height / nb_jobs;
    if (slice_size * nb_jobs < s->b_height)
        slice_size++;
    int slice_start = jobnr * slice_size;
    int slice_end = min(slice_start + slice_size, s->b_height);
    int mv_count = slice_start * s->b_width + (dir * s->b_height * s->b_width);

    for (mb_y = slice_start; mb_y < slice_end; mb_y++) {
        for (mb_x = 0; mb_x < s->b_width; mb_x++) {
            do {
                for (mb_y = slice_start; mb_y < slice_end; mb_y++)
                    for (mb_x = 0; mb_x < s->b_width; mb_x++) {
                        const int x_mb = (mb_x << s->log2_mb_size);
                        const int y_mb = (mb_y << s->log2_mb_size);
                        int mv[2] = {x_mb, y_mb};
                        ff_me_search_tdls2(me_ctx, x_mb, y_mb, mv);
                        add_mv_data(td->mv_data + mv_count++, s->mb_size, x_mb, y_mb, mv[0], mv[1], dir);
                    }
            } while (0);
        }
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    MEContext *s = ctx->priv;
    AVMotionEstContext2 *me_ctx = &s->me_ctx;
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

    sd = av_frame_new_side_data(out, AV_FRAME_DATA_MOTION_VECTORS, 1 * s->b_count * sizeof(AVMotionVector));
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

const AVFilter ff_vf_mestimate2 = {
    .name          = "mestimate2",
    .description   = NULL_IF_CONFIG_SMALL("Generate motion vectors optimized for big object motion."),
    .priv_size     = sizeof(MEContext),
    .priv_class    = &mestimate2_class,
    .uninit        = uninit,
    .flags         = AVFILTER_FLAG_METADATA_ONLY | AVFILTER_FLAG_SLICE_THREADS,
    FILTER_INPUTS(mestimate_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
