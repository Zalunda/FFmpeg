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

typedef struct CellMetrics CellMetrics;

typedef struct CellMetrics {
    int index_column;
    int index_row;
    int x;
    int y;

    int motion_x;
    int motion_y;

    CellMetrics *neighbours[9];
    uint32_t nb_similar_nearby;
    int group_id;
} CellMetrics;

typedef struct {
    float cos_angle;
    float sin_angle;
    uint8_t length;    // normalized length 0-255
} VectorMetrics;

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

    int64_t last_pts;
    int nb_columns;
    int nb_rows;
    int nb_cells;
    CellMetrics *cells_metrics;

    VectorMetrics vector_lookup[256][256];
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

static void init_vector_lookup(ExtractMVsContext *s) {
    for(int mx = -127; mx <= 128; mx++) {
        for(int my = -127; my <= 128; my++) {
            int idx_x = mx + 127;
            int idx_y = my + 127;

            // Calculate angle (0-359)
            float angle = atan2f(my, mx) * 180.0f / M_PI;
            if(angle < 0) angle += 360.0f;


            // Calculate length (normalized to 0-255)
            float raw_length = sqrtf(mx*mx + my*my);
            uint8_t norm_length = raw_length > 255 ? 255 : (uint8_t)raw_length;

            float angle_rad = atan2f(my, mx);
            s->vector_lookup[idx_y][idx_x].cos_angle = cosf(angle_rad);
            s->vector_lookup[idx_y][idx_x].sin_angle = sinf(angle_rad);
            s->vector_lookup[idx_y][idx_x].length = norm_length;
        }
    }
}

static inline VectorMetrics get_vector_metrics(const ExtractMVsContext *s, const CellMetrics *cell) {
    int idx_x = cell->motion_x + 127;
    int idx_y = cell->motion_y + 127;
    VectorMetrics vm = s->vector_lookup[idx_x][idx_y];
    return vm;
}

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
        avio_wl16(s->io_ctx, 0);  // nb_columns
        avio_wl16(s->io_ctx, 0);  // nb_rows

        char for_future_use[24];
        avio_write(s->io_ctx, for_future_use, sizeof(for_future_use));

        init_vector_lookup(s);
    }

    return 0;
}

static inline int is_similar_motion(const ExtractMVsContext *s,
                                  const CellMetrics *cell1,
                                  const CellMetrics *cell2,
                                  int angle_accepted,
                                  int minimum_length)
{
    VectorMetrics vm1 = get_vector_metrics(s, cell1);
    VectorMetrics vm2 = get_vector_metrics(s, cell2);

    if (vm1.length < minimum_length || vm2.length < minimum_length)
        return 0;

    // Dot product gives us cosine of angle between vectors
    float dot = vm1.cos_angle * vm2.cos_angle +
                vm1.sin_angle * vm2.sin_angle;

    // Convert angle_accepted to cosine threshold
    float cos_threshold = cosf(angle_accepted * M_PI / 180.0f);

    return dot >= cos_threshold;
}

static uint32_t compute_nb_similar_nearby(const ExtractMVsContext *s, const CellMetrics *cell, int angle_accepted, int minimum_length) {
    uint32_t similar_count = 0;

    // Iterate through the neighbour array until we hit NULL
    for (int i = 0; cell->neighbours[i] != NULL; i++) {
        if (is_similar_motion(s, cell, cell->neighbours[i], angle_accepted, minimum_length)) {
            similar_count++;
        }
    }
    if (similar_count > 5)
    {
        int k = 0;
    }
    return similar_count;
}

typedef struct {
    int group_id;
    int nb_cells;
    double sum_cos;
    double sum_sin;
    int total_x;
    int total_y;
} GroupStats;

static CellMetrics *find_best_initial_cell(ExtractMVsContext *s, int angle_accepted, int minimum_length) {
    int max_similar = 5;
    float max_coherence = -1.0f;  // Changed from min_angle_range
    CellMetrics *best_cell = NULL;
    int best_length = 0;

    for (int i = 0; i < s->nb_cells; i++) {
        CellMetrics *current = &s->cells_metrics[i];
        if (current->group_id != 0 || current->nb_similar_nearby < max_similar)
            continue;

        VectorMetrics current_vm = get_vector_metrics(s, current);
        if (current_vm.length < minimum_length)
            continue;

        // Sum of vector components for coherence calculation
        float sum_cos = current_vm.cos_angle;
        float sum_sin = current_vm.sin_angle;
        int nb_acceptable = 1;  // Include current cell

        for (int j = 0; current->neighbours[j] != NULL; j++) {
            if (current->group_id == 0 &&
                is_similar_motion(s, current, current->neighbours[j], angle_accepted, minimum_length)) {
                VectorMetrics vm = get_vector_metrics(s, current->neighbours[j]);
                sum_cos += vm.cos_angle;
                sum_sin += vm.sin_angle;
                nb_acceptable++;
            }
        }

        // Calculate coherence as the length of the mean vector
        float coherence = sqrtf(sum_cos * sum_cos + sum_sin * sum_sin) / nb_acceptable;

        if (nb_acceptable > max_similar ||
            (nb_acceptable == max_similar && coherence > max_coherence)) {
            max_similar = nb_acceptable;
            max_coherence = coherence;
            best_cell = current;
            best_length = current_vm.length;
        }
    }

    return best_cell;
}

static void process_group(ExtractMVsContext *s, CellMetrics *start_cell, GroupStats *stats, int minimum_length) {
    // Mark and process the start cell first
    start_cell->group_id = stats->group_id;

    VectorMetrics vm = get_vector_metrics(s, start_cell);
    stats->nb_cells++;
    stats->sum_cos += vm.cos_angle;
    stats->sum_sin += vm.sin_angle;
    stats->total_x += start_cell->x;
    stats->total_y += start_cell->y;

    // Calculate angle boundaries
    float mean_angle_rad = atan2f(stats->sum_sin, stats->sum_cos);
    if (mean_angle_rad < -M_PI) mean_angle_rad += 2 * M_PI;
    else if (mean_angle_rad > M_PI) mean_angle_rad -= 2 * M_PI;
    float angle_range_rad = 60.0f * M_PI / 180.0f;

    float cos_min = cosf(mean_angle_rad - angle_range_rad);
    float sin_min = sinf(mean_angle_rad - angle_range_rad);
    float cos_max = cosf(mean_angle_rad + angle_range_rad);
    float sin_max = sinf(mean_angle_rad + angle_range_rad);

    // Collect valid neighbors first (max 8 neighbors as seen in context)
    CellMetrics *valid_neighbors[9] = {NULL};  // 9th element as sentinel
    int valid_count = 0;

    for (int i = 0; start_cell->neighbours[i] != NULL; i++) {
        CellMetrics *neighbor = start_cell->neighbours[i];

        if (neighbor->group_id != 0)
            continue;

        VectorMetrics vm_n = get_vector_metrics(s, neighbor);
        if (vm_n.length < minimum_length)
            continue;

        float cross1 = vm_n.cos_angle * sin_min - vm_n.sin_angle * cos_min;
        float cross2 = vm_n.cos_angle * sin_max - vm_n.sin_angle * cos_max;

        if (cross1 * cross2 <= 0) {
            // Mark the neighbor immediately
            neighbor->group_id = stats->group_id;
            valid_neighbors[valid_count++] = neighbor;
            if (valid_count >= 8) break;  // Safety check
        }
    }
    valid_neighbors[valid_count] = NULL;  // Null terminator

    // Now recursively process the pre-validated neighbors
    for (int i = 0; valid_neighbors[i] != NULL; i++) {
        process_group(s, valid_neighbors[i], stats, minimum_length);
    }
}

static void draw_circle(AVFrame *frame, float cx, float cy, float radius, uint8_t color[4]) {
    uint8_t *data = frame->data[0];
    int linesize = frame->linesize[0];
    int width = frame->width;
    int height = frame->height;

    // Use Bresenham's circle algorithm
    int x = radius;
    int y = 0;
    int err = 0;

    while (x >= y) {
        // Draw 8 points of the circle
        int points[][2] = {
            {cx + x, cy + y}, {cx + y, cy + x},
            {cx - y, cy + x}, {cx - x, cy + y},
            {cx - x, cy - y}, {cx - y, cy - x},
            {cx + y, cy - x}, {cx + x, cy - y}
        };

        for (int i = 0; i < 8; i++) {
            int px = points[i][0];
            int py = points[i][1];
            if (px >= 0 && px < width && py >= 0 && py < height) {
                data[py * linesize + px] = color[0]; // Using first color component
            }
        }

        y += 1;
        err += 1 + 2*y;
        if (2*(err-x) + 1 > 0) {
            x -= 1;
            err += 1 - 2*x;
        }
    }
}

void draw_motion_group(AVFrame *frame, GroupStats stats, int mb_size, uint8_t color[4]) {
    float center_x = stats.total_x / stats.nb_cells;
    float center_y = stats.total_y / stats.nb_cells;
    float mean_angle = atan2f(stats.sum_sin, stats.sum_cos) * 180.0f / M_PI;
    if (mean_angle < 0) mean_angle += 360.0f;

    // Calculate size
    float area = stats.nb_cells * mb_size * mb_size;
    float radius = sqrtf(area / M_PI);

    // Draw main circle
    draw_circle(frame, center_x, center_y, radius, color);

    // Draw direction indicator
    // float arrow_length = radius * 1.5f;
    // float end_x = center_x + cosf(mean_angle * M_PI / 180.0f) * arrow_length;
    // float end_y = center_y + sinf(mean_angle * M_PI / 180.0f) * arrow_length;
    // draw_arrow(frame, center_x, center_y, end_x, end_y, color);

    // Optionally draw text with stats
    // char text[64];
    // snprintf(text, sizeof(text), "n=%d a=%.1f", stats.nb_cells, mean_angle);
    // draw_text(frame, center_x, center_y - radius, text, color);
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
            s->nb_columns = frame->width / s->mb_size;
            s->nb_rows = frame->height / s->mb_size;
            s->nb_cells = s->nb_rows * s->nb_columns;

            // Allocate the motion vector cells
            s->cells_metrics = av_calloc(s->nb_cells, sizeof(CellMetrics));

            for (int i = 0; i < s->nb_cells; i++) {
                CellMetrics *current = &s->cells_metrics[i];
                current->index_row = i / s->nb_columns;
                current->index_column = i % s->nb_columns;
                current->x = current->index_column * s->mb_size;
                current->y = current->index_row * s->mb_size;
                current->motion_x = 0;
                current->motion_y = 0;

                int index = 0;
                for (int orow = -1; orow <= 1; orow++) {
                    for (int ocolumn = -1; ocolumn <= 1; ocolumn++) {
                        if (ocolumn == 0 && orow == 0)
                            continue;
                        int nrow = current->index_row + orow;
                        int ncolumn = current->index_column + ocolumn;
                        if (ncolumn >= 0 && ncolumn < s->nb_columns &&
                            nrow >= 0 && nrow < s->nb_rows) {
                            current->neighbours[index++] = &s->cells_metrics[nrow * s->nb_columns + ncolumn];
                        }
                    }
                }
                current->neighbours[index] = NULL;
            }
        }

        // Clear the motion vector array before each frame
        for (int i = 0; i < s->nb_cells; i++) {
            CellMetrics *current = &s->cells_metrics[i];
            current->motion_x = 0;
            current->motion_y = 0;
        }

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

                int start_index_column = block_start_x / s->mb_size;
                int start_index_row = block_start_y / s->mb_size;
                int end_index_column = (block_end_x + s->mb_size - 1) / s->mb_size;
                int end_index_row = (block_end_y + s->mb_size - 1) / s->mb_size;

                int motion_x = mvs[i].dst_x - mvs[i].src_x;
                int motion_y = mvs[i].dst_y - mvs[i].src_y;

                for (int row = start_index_row; row < end_index_row; row++) {
                    for (int column = start_index_column; column < end_index_column; column++) {
                        if (column >= 0 && column < s->nb_columns && row >= 0 && row < s->nb_rows) {
                            int overlap_start_x = FFMAX(block_start_x, column * s->mb_size);
                            int overlap_start_y = FFMAX(block_start_y, row * s->mb_size);
                            int overlap_end_x = FFMIN(block_end_x, (column + 1) * s->mb_size);
                            int overlap_end_y = FFMIN(block_end_y, (row + 1) * s->mb_size);

                            int overlap_width = overlap_end_x - overlap_start_x;
                            int overlap_height = overlap_end_y - overlap_start_y;

                            int overlap_area = overlap_width * overlap_height;

                            CellMetrics *mva = &s->cells_metrics[row * s->nb_columns + column];
                            mva->motion_x += (int)motion_x * overlap_area;
                            mva->motion_y += (int)motion_y * overlap_area;
                        }
                    }
                }
            }

            int angle_accepted = 100;
            int minimum_speed = 2;
            int mb_size_squared = s->mb_size * s->mb_size;
            for (int i = 0; i < s->nb_cells; i++) {
                CellMetrics *current = &s->cells_metrics[i];
                current->motion_x = current->motion_x / mb_size_squared;
                current->motion_y = current->motion_y / mb_size_squared;
                current->nb_similar_nearby = compute_nb_similar_nearby(s, current, angle_accepted, minimum_speed);
            }

            int current_group_id = 1;
            int minimum_length = 3;
            CellMetrics *best_cell;
            while (best_cell = find_best_initial_cell(s, angle_accepted, minimum_length))
            {
                GroupStats stats = {
                    .group_id = current_group_id++,
                    .sum_cos = 0,
                    .sum_sin = 0,
                    .nb_cells = 0,
                    .total_x = 0,
                    .total_y = 0
                };
                process_group(s, best_cell, &stats, minimum_length);

                // Now you can draw your circle at:
                float center_x = stats.total_x / stats.nb_cells;
                float center_y = stats.total_y / stats.nb_cells;
                float mean_angle = atan2f(stats.sum_sin, stats.sum_cos) * 180.0f / M_PI;
                if (mean_angle < 0) mean_angle += 360.0f;

                uint8_t color[4];
                color[0] = (stats.group_id * 37) % 256;  // R
                color[1] = (stats.group_id * 73) % 256;  // G
                color[2] = (stats.group_id * 151) % 256; // B
                color[3] = 255;  // Alpha

                draw_motion_group(frame, stats, s->mb_size, color);
            }
        }

        // Write the motion vector array
        for (int index = 0; index < s-> nb_cells; index++)
        {
            CellMetrics *mva = &s->cells_metrics[index];
            avio_w8(s->io_ctx, mva->motion_x);
            avio_w8(s->io_ctx, mva->motion_y);
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
    if (s->cells_metrics) {
        av_free(s->cells_metrics);
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
        avio_wl16(s->io_ctx, s->nb_columns);        // nb_columns
        avio_wl16(s->io_ctx, s->nb_rows);           // nb_rows
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