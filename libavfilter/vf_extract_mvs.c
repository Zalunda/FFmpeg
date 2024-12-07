#include <libavutil/opt.h>
#include <libavutil/motion_vector.h>
#include <libavutil/mem.h>
#include <libavformat/avio.h>
#include "avfilter.h"
#include "internal.h"
#include "video.h"

typedef enum {
    FILE_FORMAT_RAW = 1,
    FILE_FORMAT_MATRIX = 2,
} FileFormat;

typedef struct {
    int motion_x;
    int motion_y;
} MotionVectorAccumulator;

typedef struct {
    const AVClass *class;
    AVFilterContext *ctx;
    char *filename;
    char *format_string;
    FileFormat format;
    AVIOContext *io_ctx;
    int frame_number;
    int mb_size;
    int video_width;
    int video_height;
    AVRational stream_time_base;
    int nb_blocks_x;
    int nb_blocks_y;
    int nb_blocks_total;
    int64_t last_pts;
    MotionVectorAccumulator *motion_vectors_matrix;
} ExtractMVsContext;

#define OFFSET(x) offsetof(ExtractMVsContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption extract_mvs_options[] = {
    {"filename", "output filename", OFFSET(filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"format", "file format", OFFSET(format_string), AV_OPT_TYPE_STRING, {.str = "matrix"}, 0, 0, FLAGS},
    {"mb_size", "block size", OFFSET(mb_size), AV_OPT_TYPE_INT, {.i64 = 16}, 8, 64, FLAGS},
    {NULL}
};

AVFILTER_DEFINE_CLASS(extract_mvs);

static av_cold int init(AVFilterContext *ctx)
{
    ExtractMVsContext *s = ctx->priv;
    int ret = avio_open(&s->io_ctx, s->filename, AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Could not open file '%s' for writing.\n", s->filename);
        return ret;
    }

    s->format = FILE_FORMAT_MATRIX;

    avio_w8(s->io_ctx, 'F');
    avio_w8(s->io_ctx, 'M');
    avio_w8(s->io_ctx, 'V');
    avio_w8(s->io_ctx, 'S');
    avio_w8(s->io_ctx, (int8_t)s->format);

    if (s->format == FILE_FORMAT_RAW) {
        int version_raw = 1;
        avio_w8(s->io_ctx, version_raw);
    } else if (s->format == FILE_FORMAT_MATRIX) {
        int version_matrix = 1;
        avio_w8(s->io_ctx, version_matrix);

        // Write dummy values for video width, height, and nb_frames
        avio_wl16(s->io_ctx, 0);  // video_width
        avio_wl16(s->io_ctx, 0);  // video_height
        avio_wl32(s->io_ctx, 0);  // video_duration
        avio_wl32(s->io_ctx, 0);  // video_frame_rate
        avio_wl32(s->io_ctx, 0);  // nb_frames
        avio_wl16(s->io_ctx, 0);  // block_size_x
        avio_wl16(s->io_ctx, 0);  // block_size_y
        avio_wl16(s->io_ctx, 0);  // sensor_size_x
        avio_wl16(s->io_ctx, 0);  // sensor_size_y
        avio_wl16(s->io_ctx, 0);  // nb_blocks_x
        avio_wl16(s->io_ctx, 0);  // nb_blocks_y

        char for_future_use[24];
        avio_write(s->io_ctx, for_future_use, sizeof(for_future_use));
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    ExtractMVsContext *s = ctx->priv;

    AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    s->frame_number++;
    s->last_pts = frame->pts;

    if (s->format == FILE_FORMAT_RAW) {
        if (sd) {
            const AVMotionVector* mvs = (const AVMotionVector*)sd->data;
            int nb_mvs = (int)sd->size / sizeof(*mvs);

            for (int i = 0; i < nb_mvs; i++) {
                avio_wl32(s->io_ctx, s->frame_number);
                avio_w8(s->io_ctx, mvs[i].source);
                avio_w8(s->io_ctx, mvs[i].w);
                avio_w8(s->io_ctx, mvs[i].h);
                avio_wl16(s->io_ctx, mvs[i].src_x);
                avio_wl16(s->io_ctx, mvs[i].src_y);
                avio_wl16(s->io_ctx, mvs[i].dst_x);
                avio_wl16(s->io_ctx, mvs[i].dst_y);
                avio_wl64(s->io_ctx, mvs[i].flags);
            }
        }
    } else if (s->format == FILE_FORMAT_MATRIX) {
        // Update video width and height if not set
        if (s->video_width == 0 || s->video_height == 0) {
            s->video_width = frame->width;
            s->video_height = frame->height;
            s->stream_time_base = inlink->time_base;
            s->nb_blocks_x = frame->width / s->mb_size;
            s->nb_blocks_y = frame->height / s->mb_size;
            s->nb_blocks_total = s->nb_blocks_y * s->nb_blocks_x;

            // Allocate the motion vector array
            s->motion_vectors_matrix = av_calloc(s->nb_blocks_total, sizeof(MotionVectorAccumulator));
        }

        // Clear the motion vector array before each frame
        memset(s->motion_vectors_matrix, 0, s->nb_blocks_total * sizeof(MotionVectorAccumulator));

        // Get the frame timestamp in milliseconds
        int64_t timestamp_ms = av_rescale_q(frame->pts, inlink->time_base, av_make_q(1, 1000));
        
        // Write frame number and frame time
        avio_wl32(s->io_ctx, s->frame_number);
        avio_wl32(s->io_ctx, timestamp_ms);
        char for_future_use[12];
        avio_write(s->io_ctx, for_future_use, sizeof(for_future_use));

        if (sd) {
            const AVMotionVector* mvs = (const AVMotionVector*)sd->data;
            int nb_mvs = (int)sd->size / sizeof(*mvs);

            for (int i = 0; i < nb_mvs; i++) {
                int block_center_x = mvs[i].dst_x;
                int block_center_y = mvs[i].dst_y;
                int block_start_x = block_center_x - s->mb_size / 2 - 1;
                int block_start_y = block_center_y - s->mb_size / 2 - 1;
                int block_end_x = block_start_x + s->mb_size;
                int block_end_y = block_start_y + s->mb_size;

                int start_block_x = block_start_x / s->mb_size;
                int start_block_y = block_start_y / s->mb_size;
                int end_block_x = (block_end_x + s->mb_size - 1) / s->mb_size;
                int end_block_y = (block_end_y + s->mb_size - 1) / s->mb_size;

                int motion_x = mvs[i].dst_x - mvs[i].src_x;
                int motion_y = mvs[i].dst_y - mvs[i].src_y;

                for (int by = start_block_y; by < end_block_y; by++) {
                    for (int bx = start_block_x; bx < end_block_x; bx++) {
                        if (bx >= 0 && bx < s->nb_blocks_x && by >= 0 && by < s->nb_blocks_y) {
                            int overlap_start_x = FFMAX(block_start_x, bx * s->mb_size);
                            int overlap_start_y = FFMAX(block_start_y, by * s->mb_size);
                            int overlap_end_x = FFMIN(block_end_x, (bx + 1) * s->mb_size);
                            int overlap_end_y = FFMIN(block_end_y, (by + 1) * s->mb_size);

                            int overlap_width = overlap_end_x - overlap_start_x;
                            int overlap_height = overlap_end_y - overlap_start_y;

                            int overlap_area = overlap_width * overlap_height;

                            MotionVectorAccumulator *mva = &s->motion_vectors_matrix[by * s->nb_blocks_x + bx];
                            mva->motion_x += (int)motion_x * overlap_area;
                            mva->motion_y += (int)motion_y * overlap_area;
                        }
                    }
                }
            }
        }

        // Write the motion vector array
        int mb_size_squared = s->mb_size * s->mb_size;
        for (int index = 0; index < s-> nb_blocks_total; index++)
        {
            MotionVectorAccumulator *mva = &s->motion_vectors_matrix[index];
            int reduced_motion_x = mva->motion_x / mb_size_squared;
            int reduced_motion_y = mva->motion_y / mb_size_squared;
            avio_w8(s->io_ctx, reduced_motion_x);
            avio_w8(s->io_ctx, reduced_motion_y);
        }
    }

    return ff_filter_frame(ctx->outputs[0], frame);
}

static int normalize_frame_rate(int computed_rate) {
    // Common frame rates * 100
    const int known_rates[] = {
        2397, 2400,   // 23.97, 24
        2997, 3000,   // 29.97, 30
        5994, 6000,   // 59.94, 60
        11988, 12000  // 119.88, 120
    };
    
    const int tolerance = 1;
    
    for (int i = 0; i < sizeof(known_rates) / sizeof(known_rates[0]); i++) {
        if (abs(computed_rate - known_rates[i]) <= tolerance) {
            return known_rates[i];
        }
    }
    
    return computed_rate;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ExtractMVsContext *s = ctx->priv;

    // Free the motion vector array
    if (s->motion_vectors_matrix) {
        av_free(s->motion_vectors_matrix);
    }

    if (s->format == FILE_FORMAT_MATRIX && s->frame_number > 0) {

        // Compute the video frame rate per 100 frames
        int64_t video_duration_ms = av_rescale_q(s->last_pts, s->stream_time_base, av_make_q(1, 1000));
        int video_frame_rate = (int)round((float)100 * 1000 / ((float)video_duration_ms / (s->frame_number - 1)));  // "frame_number - 1" because last_pts doesn't include the duration of the last frame
        int normalized = normalize_frame_rate(video_frame_rate);

        // Seek back and update the values in the file
        avio_seek(s->io_ctx, 6, SEEK_SET);
        avio_wl16(s->io_ctx, s->video_width);       // video_width
        avio_wl16(s->io_ctx, s->video_height);      // video_height
        avio_wl32(s->io_ctx, video_duration_ms);    // video_duration
        avio_wl32(s->io_ctx, video_frame_rate);     // video_frame_rate per 100 frames (ex. 5994 for 59.94)
        avio_wl32(s->io_ctx, s->frame_number);      // nb_frames
        avio_wl16(s->io_ctx, s->mb_size);           // block_size_x
        avio_wl16(s->io_ctx, s->mb_size);           // block_size_y
        avio_wl16(s->io_ctx, s->mb_size);           // sensor_size_x
        avio_wl16(s->io_ctx, s->mb_size);           // sensor_size_y
        avio_wl16(s->io_ctx, s->nb_blocks_x);       // nb_blocks_x
        avio_wl16(s->io_ctx, s->nb_blocks_y);       // nb_blocks_y
    }

    avio_close(s->io_ctx);
}

static const AVFilterPad extract_mvs_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    }
};

const AVFilter ff_vf_extract_mvs = {
    .name          = "extract_mvs",
    .description   = NULL_IF_CONFIG_SMALL("Extract motion vectors from a video."),
    .priv_size     = sizeof(ExtractMVsContext),
    .priv_class    = &extract_mvs_class,
    .init          = init,
    .uninit        = uninit,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    FILTER_INPUTS(extract_mvs_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad)
};