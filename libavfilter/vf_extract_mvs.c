#include <libavutil/opt.h>
#include <libavutil/motion_vector.h>
#include <libavutil/mem.h>
#include <libavformat/avio.h>
#include "avfilter.h"
#include "video.h"
#include <math.h> // For atan2f, sqrtf, cosf, sinf, fabsf, roundf
#include <limits.h> // For INT_MAX, INT_MIN

typedef enum {
    FILE_FORMAT_RAW = 1,
    FILE_FORMAT_MATRIX = 2,
} FileFormat;

typedef struct {
    uint16_t angle;
    uint8_t length;
} VectorMetrics;

typedef struct CellMetrics CellMetrics;

typedef struct {
    int is_final;
    int group_id;
    int initial_angle;
    int real_angle;
    int nb_cells;
    int total_x;
    int total_y;
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    int total_motion_x;
    int total_motion_y;

    uint64_t bucket_bits;
} GroupStats;

typedef struct CellMetrics {
    int index_column;
    int index_row;
    int x;
    int y;
    int random_value;
    CellMetrics *neighbours[9];

    int is_final;
    int motion_x;
    int motion_y;
    VectorMetrics vm;
    int bucket_index;
    uint64_t bucket_bit;
    GroupStats *in_group;
} CellMetrics;

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
    CellMetrics **cells_metrics_sortable;
    CellMetrics **cells_metrics_queue;
    GroupStats *groups_stats;
    GroupStats **groups_stats_sortable;

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

// Function Prototypes (forward declarations for C90 compliance)
static void init_vector_lookup(ExtractMVsContext *s);
static inline VectorMetrics get_vector_metrics(const ExtractMVsContext *s, const int motion_x, const int motion_y);
static int init(AVFilterContext *ctx);
static void draw_circle(AVFrame *frame, float cx, float cy, float radius, uint8_t color[4]);
static int clip_line(int *sx, int *sy, int *ex, int *ey, int maxx);
static void draw_line(uint8_t *buf, int sx, int sy, int ex, int ey,
                     int w, int h, ptrdiff_t stride, int color);
static void draw_arrow(AVFrame *frame, float sx, float sy, float ex, float ey, uint8_t color[4]);
static void draw_motion_group(AVFrame *frame, GroupStats *group, int mb_size, uint8_t color[4]); // <-- MISSING PROTOTYPE
static int compare_cells(const void *a, const void *b);
static int compare_groups(const void *a, const void *b);
static uint64_t step2_find_best_angle(CellMetrics **cells, int nb_cells, int nb_buckets_on_each_side, int *best_angle);
static void step3_process_group(CellMetrics **queue, int cell_size, GroupStats *group, CellMetrics *starting_cell);
static int step3_create_groups(CellMetrics **queue, int cell_size, int *current_group_id, GroupStats **groups, uint64_t bucket_bits, int initial_angle, CellMetrics **cells, int nb_cells, int min_group_size);
static void postprocessing_cells(ExtractMVsContext *s, AVFrame *frame);
static int filter_frame(AVFilterLink *inlink, AVFrame *frame);
static int normalize_frame_rate(int computed_rate);
static void uninit(AVFilterContext *ctx);


static void init_vector_lookup(ExtractMVsContext *s) {
    int mx, my;
    int idx_x, idx_y;
    float angle, raw_length;
    uint8_t norm_length;

    for(mx = -127; mx <= 128; mx++) {
        for(my = -127; my <= 128; my++) {
            idx_x = mx + 127;
            idx_y = my + 127;

            // Calculate angle (0-359)
            angle = atan2f(my, mx) * 180.0f / M_PI;
            if(angle < 0) angle += 360.0f;

            // Calculate length (normalized to 0-255)
            raw_length = sqrtf((float)mx*mx + (float)my*my);
            norm_length = raw_length > 255 ? 255 : (uint8_t)raw_length;

            s->vector_lookup[idx_x][idx_y].angle = (uint16_t)angle;
            s->vector_lookup[idx_x][idx_y].length = norm_length;
        }
    }
}

static inline VectorMetrics get_vector_metrics(const ExtractMVsContext *s, const int motion_x, const int motion_y) {
    int idx_x = motion_x + 127;
    int idx_y = motion_y + 127;
    return s->vector_lookup[idx_x][idx_y];
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

static void draw_circle(AVFrame *frame, float cx, float cy, float radius, uint8_t color[4]) {
    uint8_t *data = frame->data[0];
    int linesize = frame->linesize[0];
    int width = frame->width;
    int height = frame->height;

    // Use Bresenham's circle algorithm
    int x = (int)radius;
    int y = 0;
    int err = 0;

    int i, dx, dy, px, py;

    while (x >= y) {
        // Draw 8 points of the circle with thickness
        int points[][2] = {
            {(int)(cx + x), (int)(cy + y)}, {(int)(cx + y), (int)(cy + x)},
            {(int)(cx - y), (int)(cy + x)}, {(int)(cx - x), (int)(cy + y)},
            {(int)(cx - x), (int)(cy - y)}, {(int)(cx - y), (int)(cy - x)},
            {(int)(cx + y), (int)(cy - x)}, {(int)(cx + x), (int)(cy - y)}
        };

        for (i = 0; i < 8; i++) {
            // Draw a 3x3 square around each point
            for (dy = -1; dy <= 2; dy++) {
                for (dx = -1; dx <= 2; dx++) {
                    px = points[i][0] + dx;
                    py = points[i][1] + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        data[py * linesize + px] = color[0];
                    }
                }
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

static int clip_line(int *sx, int *sy, int *ex, int *ey, int maxx)
{
    int tmp;
    if (*sx > *ex) {
        // Swap points if start x is greater than end x
        tmp = *sx; *sx = *ex; *ex = tmp;
        tmp = *sy; *sy = *ey; *ey = tmp;
    }

    // Clip against x = 0
    if (*sx < 0) {
        if (*ex < 0)
            return 1;  // Line completely outside
        *sy = *sy + (*ey - *sy) * (0 - *sx) / (*ex - *sx);
        *sx = 0;
    }

    // Clip against x = maxx
    if (*ex > maxx) {
        if (*sx > maxx)
            return 1;  // Line completely outside
        *ey = *sy + (*ey - *sy) * (maxx - *sx) / (*ex - *sx);
        *ex = maxx;
    }

    return 0;  // Line (partially) inside
}

static void draw_line(uint8_t *buf, int sx, int sy, int ex, int ey,
                     int w, int h, ptrdiff_t stride, int color)
{
    // First clip the line to the frame boundaries
    if (clip_line(&sx, &sy, &ex, &ey, w - 1))
        return;
    if (clip_line(&sy, &sx, &ey, &ex, h - 1))
        return;

    // Bresenham algorithm implementation
    int dx = abs(ex - sx);
    int dy = abs(ey - sy);
    int step_x = sx < ex ? 1 : -1;
    int step_y = sy < ey ? 1 : -1;
    int err = dx - dy;
    int e2;

    while (1) {
        // Draw the current pixel
        if (sy >= 0 && sy < h && sx >= 0 && sx < w) { // Added bounds check
            buf[sy * stride + sx] = color;
        }

        // Check if we reached the end point
        if (sx == ex && sy == ey)
            break;

        e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            sx += step_x;
        }
        if (e2 < dx) {
            err += dx;
            sy += step_y;
        }
    }
}

static void draw_arrow(AVFrame *frame, float sx, float sy, float ex, float ey, uint8_t color[4]) {
    uint8_t *data = frame->data[0];
    int linesize = frame->linesize[0];
    int width = frame->width;
    int height = frame->height;


    float dx, dy, length, nx, ny;
    int i;
    float offset_x, offset_y;
    float start_x, start_y, end_x, end_y;
    int x0, y0, x1, y1;
    int dx_inner, dy_inner, sx_inner, sy_inner, err_inner;
    int e2;
    float angle, arrow_length, arrow_angle;
    float arrow_x1, arrow_y1, arrow_x2, arrow_y2;

    // Draw main line (5 pixels wide)
    dx = ex - sx;
    dy = ey - sy;
    length = sqrtf(dx * dx + dy * dy);

    if (length == 0) return; // Avoid division by zero

    nx = -dy / length;  // normalized perpendicular vector
    ny = dx / length;

    // Draw multiple parallel lines to create thickness
    for (i = -2; i <= 2; i++) {
        offset_x = nx * i;
        offset_y = ny * i;

        start_x = sx + offset_x;
        start_y = sy + offset_y;
        end_x = ex + offset_x;
        end_y = ey + offset_y;

        // Draw the line using Bresenham-like algorithm from draw_line
        x0 = (int)roundf(start_x);
        y0 = (int)roundf(start_y);
        x1 = (int)roundf(end_x);
        y1 = (int)roundf(end_y);

        dx_inner = abs(x1 - x0);
        dy_inner = abs(y1 - y0);
        sx_inner = x0 < x1 ? 1 : -1;
        sy_inner = y0 < y1 ? 1 : -1;
        err_inner = dx_inner - dy_inner;

        while (1) {
            if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                data[y0 * linesize + x0] = color[0];
            }

            if (x0 == x1 && y0 == y1) break;

            e2 = 2 * err_inner;
            if (e2 > -dy_inner) {
                err_inner -= dy_inner;
                x0 += sx_inner;
            }
            if (e2 < dx_inner) {
                err_inner += dx_inner;
                y0 += sy_inner;
            }
        }
    }

    // Draw arrowhead
    angle = atan2f(dy, dx);
    arrow_length = 20.0f;
    arrow_angle = M_PI / 6.0f; // 30 degrees

    // Draw left side of arrowhead
    arrow_x1 = ex - arrow_length * cosf(angle + arrow_angle);
    arrow_y1 = ey - arrow_length * sinf(angle + arrow_angle);
    draw_line(frame->data[0], (int)roundf(ex), (int)roundf(ey), (int)roundf(arrow_x1), (int)roundf(arrow_y1), width, height, linesize, color[0]);

    // Draw right side of arrowhead
    arrow_x2 = ex - arrow_length * cosf(angle - arrow_angle);
    arrow_y2 = ey - arrow_length * sinf(angle - arrow_angle);
    draw_line(frame->data[0], (int)roundf(ex), (int)roundf(ey), (int)roundf(arrow_x2), (int)roundf(arrow_y2), width, height, linesize, color[0]);
}

void draw_motion_group(AVFrame *frame, GroupStats *group, int mb_size, uint8_t color[4]) {

    float center_x, center_y;
    float area, radius;
    int i;
    int min_group_size; // Declare min_group_size here
    float arrow_length, end_x, end_y;

    center_x = (float)group->total_x / group->nb_cells;
    center_y = (float)group->total_y / group->nb_cells;

    // Calculate size
    area = (float)group->nb_cells * mb_size * mb_size;
    radius = sqrtf(area / M_PI);

    // Draw bounding rectangle with 5-pixel thick lines
    // Top line
    for(i = -2; i <= 2; i++) {
        draw_line(frame->data[0],
                  group->min_x, group->min_y + i,
                  group->max_x, group->min_y + i,
                  frame->width, frame->height,
                  frame->linesize[0], color[0]);
    }

    // Bottom line
    for(i = -2; i <= 2; i++) {
        draw_line(frame->data[0],
                  group->min_x, group->max_y + i,
                  group->max_x, group->max_y + i,
                  frame->width, frame->height,
                  frame->linesize[0], color[0]);
    }

    // Left line
    for(i = -2; i <= 2; i++) {
        draw_line(frame->data[0],
                  group->min_x + i, group->min_y,
                  group->min_x + i, group->max_y,
                  frame->width, frame->height,
                  frame->linesize[0], color[0]);
    }

    // Right line
    for(i = -2; i <= 2; i++) {
        draw_line(frame->data[0],
                  group->max_x + i, group->min_y,
                  group->max_x + i, group->max_y,
                  frame->width, frame->height,
                  frame->linesize[0], color[0]);
    }

    // Draw direction indicator
    min_group_size = 5;
    if (center_x < group->min_x && group->nb_cells < min_group_size)
    {
        // Removed `int i = 0;` - it was unused dead code
    }
    if (center_x > group->max_x && group->nb_cells < min_group_size)
    {
        // Removed `int i = 0;` - it was unused dead code
    }
    if (center_y < group->min_y && group->nb_cells < min_group_size)
    {
        // Removed `int i = 0;` - it was unused dead code
    }
    if (center_y > group->max_y && group->nb_cells < min_group_size)
    {
        // Removed `int i = 0;` - it was unused dead code
    }
    arrow_length = radius;
    end_x = center_x + cosf(group->real_angle * (float)M_PI / 180.0f) * arrow_length;
    end_y = center_y + sinf(group->real_angle * (float)M_PI / 180.0f) * arrow_length;
    draw_arrow(frame, center_x, center_y, end_x, end_y, color);

    // Optionally draw text with stats
    // char text[64];
    // snprintf(text, sizeof(text), "n=%d a=%.1f", stats.nb_cells, mean_angle);
    // draw_text(frame, center_x, center_y - radius, text, color);
}

// We'll need this as a global or static variable in the filter context
static uint64_t g_best_bucket_bits;

// Comparison function for qsort
static int compare_cells(const void *a, const void *b) {
    const CellMetrics *cell1 = *(const CellMetrics **)a;
    const CellMetrics *cell2 = *(const CellMetrics **)b;

    // First priority: is_final at the start
    if (cell1->is_final != cell2->is_final)
        return cell2->is_final - cell1->is_final;

    // Second priority: cells with matching bucket_bit
    int cell1_matches = (cell1->bucket_bit & g_best_bucket_bits) != 0;
    int cell2_matches = (cell2->bucket_bit & g_best_bucket_bits) != 0;
    if (cell1_matches != cell2_matches)
        return cell2_matches - cell1_matches;

    return cell1->random_value - cell2->random_value;
}

// Comparison function for group sorting
static int compare_groups(const void *a, const void *b) {
    const GroupStats *group1 = *(const GroupStats **)a;
    const GroupStats *group2 = *(const GroupStats **)b;

    if (group1->is_final != group2->is_final)
        return group1->is_final - group2->is_final;

    // Sort by number of cells (descending order)
    return group2->nb_cells - group1->nb_cells;
}

static uint64_t step2_find_best_angle(
    CellMetrics **cells,
    int nb_cells,
    int nb_buckets_on_each_side,
    int *best_angle) {

    // Compute the histogram of angles
    int histogram[36] = {0, };
    int i;
    int bucket_total, bucket_bits, bucket_index;
    int di;

    for (i = 0; i < nb_cells; i++) {
        CellMetrics *current = cells[i];
        histogram[current->bucket_index]++;
    }

    // Find the best range of buckets (the one that include the most vector)
    int best_bucket_total = 0;
    *best_angle = 0;
    uint64_t best_bucket_bits = 0;

    for (i = 0; i < 36; i++) {
        bucket_total = 0;
        bucket_bits = 0;
        for (di = -nb_buckets_on_each_side; di <= nb_buckets_on_each_side; di++) {
            bucket_index = (i + di + 36) % 36;
            bucket_total += histogram[bucket_index];
            bucket_bits |= ((uint64_t)1 << bucket_index);
        }
        if (bucket_total > best_bucket_total) {
            best_bucket_total = bucket_total;
            best_bucket_bits = bucket_bits;
            *best_angle = i * 10;
        }
    }

    return best_bucket_bits;
}

static void step3_process_group(
    CellMetrics **queue,
    int cell_size,
    GroupStats *group,
    CellMetrics *starting_cell) {

    // Use a queue to store cells to process
    int queue_start = 0;
    int queue_end = 0;
    CellMetrics *current_cell;
    CellMetrics *neighbor;
    int i;

    // Add the start cell to the queue
    queue[queue_end++] = starting_cell;
    starting_cell->in_group = group;

    while (queue_start < queue_end) {
        current_cell = queue[queue_start++];

        // Update group statistics
        group->nb_cells++;
        group->total_x += current_cell->x;
        group->total_y += current_cell->y;
        group->total_motion_x += current_cell->motion_x;
        group->total_motion_y += current_cell->motion_y;
        group->min_x = FFMIN(group->min_x, current_cell->x);
        group->max_x = FFMAX(group->max_x, current_cell->x + cell_size);
        group->min_y = FFMIN(group->min_y, current_cell->y);
        group->max_y = FFMAX(group->max_y, current_cell->y + cell_size);

        // Check all neighbors
        for (i = 0; current_cell->neighbours[i] != NULL; i++) {
            neighbor = current_cell->neighbours[i];

            // Skip if already processed or being processed
            if (neighbor->is_final || neighbor->in_group)
                continue;

            // Check if neighbor's motion matches our bucket criteria
            if ((neighbor->bucket_bit & group->bucket_bits) == 0)
                continue;

            // Add to queue
            neighbor->in_group = group;
            queue[queue_end++] = neighbor;
        }
    }
}

static int step3_create_groups(
    CellMetrics **queue,
    int cell_size,
    int *current_group_id,
    GroupStats **groups,
    uint64_t bucket_bits,
    int initial_angle,
    CellMetrics **cells,
    int nb_cells,
    int min_group_size)
{

    int nb_group_added = 0;
    int i;
    GroupStats *group;
    CellMetrics *starting_cell;
    CellMetrics *cell;
    float angle_diff;
    int nb_groups_to_remove;

    // Group the cells
    for (i = 0; i < nb_cells; i++) {
        starting_cell = cells[i];
        if (starting_cell->in_group)
            continue;

        group = groups[nb_group_added++];
        group->group_id = *current_group_id;
        (*current_group_id)++;
        group->is_final = 0;
        group->initial_angle = initial_angle;
        group->nb_cells = 0;
        group->total_x = 0;
        group->total_y = 0;
        group->total_motion_x = 0;
        group->total_motion_y = 0;
        group->min_x = INT_MAX;
        group->max_x = INT_MIN;
        group->min_y = INT_MAX;
        group->max_y = INT_MIN;
        group->bucket_bits = bucket_bits;
        step3_process_group(queue, cell_size, group, starting_cell);

        // Calculate real angle
        angle_diff = atan2f((float)group->total_motion_y, (float)group->total_motion_x) * 180.0f / (float)M_PI;
        if(angle_diff < 0) angle_diff += 360.0f;
        group->real_angle = (int)angle_diff;
    }

    // Sort groups by is_final & size (but is_final is 0 for all right now)
    qsort(groups, nb_group_added, sizeof(GroupStats *), compare_groups);

    // Finalize some groups
    nb_groups_to_remove = 0;
    for (i = 0; i < nb_group_added; i++) {

        group = groups[i];
        angle_diff = fabsf((float)group->real_angle - (float)group->initial_angle);
        if (angle_diff > 180.0f)
            angle_diff = 360.0f - angle_diff;

        if (((i == 0) || (angle_diff <= 30)) && (group->nb_cells > min_group_size)) {
            group->is_final = 1;
        }
        else {
            nb_groups_to_remove++;
        }
    }

    // Finalize cells that are in finalized group, cleanup the rests
    for (i = 0; i < nb_cells; i++) {
        cell = cells[i];
        if (cell->in_group && cell->in_group->is_final) {
            cell->is_final = 1;
        } else {
            cell->in_group = NULL;
        }
    }

    // Compact the groups array by sorting groups by is_final & size and then triming the nb of groups added.
    qsort(groups, nb_group_added, sizeof(GroupStats *), compare_groups);

    nb_group_added -= nb_groups_to_remove;
    return nb_group_added;
}

static void postprocessing_cells(ExtractMVsContext *s, AVFrame *frame) {
    int minimum_length = 1; // 1 or more? TO VALIDATE LATER
    int min_group_size = 3;


    int i;
    CellMetrics *current;
    CellMetrics **cells_candidates;
    int nb_cells_candidates;
    GroupStats **groups;
    int nb_groups;
    int current_group_id;
    int nb_groups_created;
    int first_cell_not_finalized;
    int best_angle;
    uint64_t best_bucket_bits;
    CellMetrics **cells_in_buckets;
    int nb_cells_in_buckets;
    GroupStats *group;
    uint8_t color[4];
    float center_x, center_y;


    for (i = 0; i < s->nb_cells; i++) {
        current = &s->cells_metrics[i];
        current->vm = get_vector_metrics(s, current->motion_x, current->motion_y);
        current->bucket_index = ((current->vm.angle - 5 + 360) % 360) / 10; // TODO validate this. I want the block 0 to be from '355 to 5'.
        current->bucket_bit = ((uint64_t)1 << current->bucket_index);
        current->is_final = current->vm.length < minimum_length;
        current->in_group = NULL;
    }

    for (i = 0; i < s->nb_cells; i++) {
        s->groups_stats_sortable[i] = &s->groups_stats[i];
    }

    cells_candidates = s->cells_metrics_sortable;
    nb_cells_candidates = s->nb_cells;

    groups = s->groups_stats_sortable;
    nb_groups = 0;
    current_group_id = 1;
    nb_groups_created = 0;

    //if (s->frame_number == 14)
    do {
        // Step 1: Eliminate the cell that are finalized
        qsort(cells_candidates, nb_cells_candidates, sizeof(CellMetrics*), compare_cells);

        first_cell_not_finalized = 0;
        while (first_cell_not_finalized < nb_cells_candidates
            && cells_candidates[first_cell_not_finalized]->is_final) {
            first_cell_not_finalized++;
        }
        cells_candidates += first_cell_not_finalized;
        nb_cells_candidates -= first_cell_not_finalized;

        // Step 2: Find the angle that include the most of the cells candidates
        best_angle = 0;
        best_bucket_bits = step2_find_best_angle(cells_candidates, nb_cells_candidates, 8, &best_angle);

        g_best_bucket_bits = best_bucket_bits;
        qsort(cells_candidates, nb_cells_candidates, sizeof(CellMetrics*), compare_cells);

        cells_in_buckets = cells_candidates;
        nb_cells_in_buckets = 0;
        while (nb_cells_in_buckets < nb_cells_candidates
            && (cells_candidates[nb_cells_in_buckets]->bucket_bit & best_bucket_bits) != 0) {
            nb_cells_in_buckets++;
        }

        // Steps 3: Create groups using the cells in the best 'buckets'
        nb_groups_created = step3_create_groups(
            s->cells_metrics_queue,
            s->mb_size,
            &current_group_id,
            &groups[nb_groups],
            best_bucket_bits,
            best_angle,
            cells_in_buckets,
            nb_cells_in_buckets,
            min_group_size);
        nb_groups += nb_groups_created;
    } while (nb_groups_created > 0);

    qsort(groups, nb_groups, sizeof(GroupStats *), compare_groups);

    for (i = 0; i < nb_groups; i++) {
        group = groups[i];

        // Now you can draw your circle at:
        center_x = (float)group->total_x / group->nb_cells;
        center_y = (float)group->total_y / group->nb_cells;

        color[0] = (uint8_t)((group->group_id * 37) % 256);  // R
        color[1] = (uint8_t)((group->group_id * 73) % 256);  // G
        color[2] = (uint8_t)((group->group_id * 151) % 256); // B
        color[3] = 255;  // Alpha
        color[0] = 255;  // R
        color[1] = 80;  // G
        color[2] = 80; // B

        draw_motion_group(frame, group, s->mb_size, color);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    ExtractMVsContext *s = ctx->priv;


    AVFrameSideData *sd;
    const AVMotionVector* mvs;
    int nb_mvs;
    int i, row, column, index;
    int block_center_x, block_center_y;
    int block_start_x, block_start_y;
    int block_end_x, block_end_y;
    int start_index_column, start_index_row;
    int end_cell_index_column, end_cell_index_row;
    int motion_x, motion_y;
    int overlap_start_x, overlap_start_y, overlap_end_x, overlap_end_y;
    int overlap_width, overlap_height, overlap_area;
    CellMetrics *mva;
    CellMetrics *current;
    int mb_size_squared;
    int orow, ocolumn, nrow, ncolumn;
    int idx;
    int64_t timestamp_ms;
    char for_future_use[12];


    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    s->frame_number++;
    s->last_pts = frame->pts;

    if (s->format == FILE_FORMAT_RAW) {
        if (sd) {
            mvs = (const AVMotionVector*)sd->data;
            nb_mvs = (int)sd->size / sizeof(*mvs);

            for (i = 0; i < nb_mvs; i++) {
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
            if (!s->cells_metrics)
                return AVERROR(ENOMEM);

            s->cells_metrics_sortable = av_malloc(sizeof(CellMetrics *) * s->nb_cells);
            if (!s->cells_metrics_sortable)
                return AVERROR(ENOMEM);

            s->cells_metrics_queue = av_malloc(sizeof(CellMetrics *) * s->nb_cells);
            if (!s->cells_metrics_queue)
                return AVERROR(ENOMEM);

            s->groups_stats = av_malloc(sizeof(GroupStats) * s->nb_cells); // TODO: Overkill number but safe
            if (!s->groups_stats)
                return AVERROR(ENOMEM);

            s->groups_stats_sortable = av_malloc(sizeof(GroupStats *) * s->nb_cells); // TODO: Overkill number but safe
            if (!s->groups_stats_sortable)
                return AVERROR(ENOMEM);


            for (i = 0; i < s->nb_cells; i++) {
                current = &s->cells_metrics[i];
                s->cells_metrics_sortable[i] = current;

                current->index_row = i / s->nb_columns;
                current->index_column = i % s->nb_columns;
                current->x = current->index_column * s->mb_size;
                current->y = current->index_row * s->mb_size;
                current->motion_x = 0;
                current->motion_y = 0;
                current->random_value = current->index_row * 31 + current->index_column * 17 % 1000;

                idx = 0;
                for (orow = -1; orow <= 1; orow++) {
                    for (ocolumn = -1; ocolumn <= 1; ocolumn++) {
                        if (ocolumn == 0 && orow == 0)
                            continue;
                        nrow = current->index_row + orow;
                        ncolumn = current->index_column + ocolumn;
                        if (ncolumn >= 0 && ncolumn < s->nb_columns &&
                            nrow >= 0 && nrow < s->nb_rows) {
                            current->neighbours[idx++] = &s->cells_metrics[nrow * s->nb_columns + ncolumn];
                        }
                    }
                }
                current->neighbours[idx] = NULL;
            }
        }

        // Get the frame timestamp in milliseconds
        timestamp_ms = av_rescale_q(frame->pts, inlink->time_base, av_make_q(1, 1000));

        // Write frame number and frame time
        avio_wl32(s->io_ctx, s->frame_number);
        avio_wl32(s->io_ctx, timestamp_ms);
        avio_write(s->io_ctx, for_future_use, sizeof(for_future_use));

        if (sd) {
            mvs = (const AVMotionVector*)sd->data;
            nb_mvs = (int)sd->size / sizeof(*mvs);

            for (i = 0; i < s->nb_cells; i++) {
                current = &s->cells_metrics[i];
                current->motion_x = 0;
                current->motion_y = 0;
            }

            // Place the motion vector in the corresponding cell
            for (i = 0; i < nb_mvs; i++) {
                block_center_x = mvs[i].dst_x;
                block_center_y = mvs[i].dst_y;
                block_start_x = block_center_x - s->mb_size / 2 - 1;
                block_start_y = block_center_y - s->mb_size / 2 - 1;
                block_end_x = block_start_x + s->mb_size;
                block_end_y = block_start_y + s->mb_size;

                start_index_column = block_start_x / s->mb_size;
                start_index_row = block_start_y / s->mb_size;
                end_cell_index_column = (block_end_x + s->mb_size - 1) / s->mb_size;
                end_cell_index_row = (block_end_y + s->mb_size - 1) / s->mb_size;

                motion_x = mvs[i].dst_x - mvs[i].src_x;
                motion_y = mvs[i].dst_y - mvs[i].src_y;

                for (row = start_index_row; row < end_cell_index_row; row++) {
                    for (column = start_index_column; column < end_cell_index_column; column++) {
                        if (column >= 0 && column < s->nb_columns && row >= 0 && row < s->nb_rows) {
                            overlap_start_x = FFMAX(block_start_x, column * s->mb_size);
                            overlap_start_y = FFMAX(block_start_y, row * s->mb_size);
                            overlap_end_x = FFMIN(block_end_x, (column + 1) * s->mb_size);
                            overlap_end_y = FFMIN(block_end_y, (row + 1) * s->mb_size);

                            overlap_width = overlap_end_x - overlap_start_x;
                            overlap_height = overlap_end_y - overlap_start_y;

                            overlap_area = overlap_width * overlap_height;

                            mva = &s->cells_metrics[row * s->nb_columns + column];
                            mva->motion_x += (int)motion_x * overlap_area;
                            mva->motion_y += (int)motion_y * overlap_area;
                        }
                    }
                }
            }

            // Write the motion vector array
            mb_size_squared = s->mb_size * s->mb_size;
            for (index = 0; index < s-> nb_cells; index++)
            {
                current = &s->cells_metrics[index];
                current->motion_x = mb_size_squared != 0 ? current->motion_x / mb_size_squared : 0;
                current->motion_y = mb_size_squared != 0 ? current->motion_y / mb_size_squared : 0;
                avio_w8(s->io_ctx, (uint8_t)current->motion_x);
                avio_w8(s->io_ctx, (uint8_t)current->motion_y);
            }

            // Un-comment to enable visualization, ensure frame->format is appropriate (e.g., AV_PIX_FMT_GRAY8)
            // postprocessing_cells(s, frame);
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
    int i;

    for (i = 0; i < (int)(sizeof(known_rates) / sizeof(known_rates[0])); i++) {
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
        s->cells_metrics = NULL;
    }
    if (s->cells_metrics_sortable) {
        av_free(s->cells_metrics_sortable);
        s->cells_metrics_sortable = NULL;
    }
    if (s->cells_metrics_queue) {
        av_free(s->cells_metrics_queue);
        s->cells_metrics_queue = NULL;
    }
    if (s->groups_stats) {
        av_free(s->groups_stats);
        s->groups_stats = NULL;
    }
    if (s->groups_stats_sortable) {
        av_free(s->groups_stats_sortable);
        s->groups_stats_sortable = NULL;
    }

    if (s->format == FILE_FORMAT_MATRIX && s->frame_number > 0) {
        // Compute the video frame rate per 100 frames
        int64_t video_duration_ms = av_rescale_q(s->last_pts, s->stream_time_base, av_make_q(1, 1000));
        int video_frame_rate = (int)roundf((float)100 * 1000.0f / ((float)video_duration_ms / (s->frame_number - 1)));  // "frame_number - 1" because last_pts doesn't include the duration of the last frame
        int normalized = normalize_frame_rate(video_frame_rate);

        // Seek back and update the values in the file
        avio_seek(s->io_ctx, 6, SEEK_SET);
        avio_wl16(s->io_ctx, (uint16_t)s->video_width);       // video_width
        avio_wl16(s->io_ctx, (uint16_t)s->video_height);      // video_height
        avio_wl32(s->io_ctx, (uint32_t)video_duration_ms);    // video_duration
        avio_wl32(s->io_ctx, (uint32_t)normalized);           // video_frame_rate per 100 frames (ex. 5994 for 59.94)
        avio_wl32(s->io_ctx, (uint32_t)s->frame_number);      // nb_frames
        avio_wl16(s->io_ctx, (uint16_t)s->mb_size);           // block_size_x
        avio_wl16(s->io_ctx, (uint16_t)s->mb_size);           // block_size_y
        avio_wl16(s->io_ctx, (uint16_t)s->mb_size);           // sensor_size_x
        avio_wl16(s->io_ctx, (uint16_t)s->mb_size);           // sensor_size_y
        avio_wl16(s->io_ctx, (uint16_t)s->nb_columns);        // nb_columns
        avio_wl16(s->io_ctx, (uint16_t)s->nb_rows);           // nb_rows
    }

    avio_close(s->io_ctx);
    s->io_ctx = NULL;
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