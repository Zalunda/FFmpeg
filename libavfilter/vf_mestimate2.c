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

typedef struct CellMetrics CellMetrics;

typedef struct CellMetrics {
    // Readonly fields
    int index_x;
    int index_y;
    int x;
    int y;
    CellMetrics *neighbours[10];

    // Fields set by the tlds algorythm (and modified afterward if needed)
    int x_mv;
    int y_mv;
    uint32_t cost;          // Store the "cost" found for this block (0 mean found a block identical in the destination)
    float similarity;       // Store the similarity metric for this block

    // Fields using during the optimisation phases
    int is_done;
    int nb_next_frame_pixels;
    uint32_t nb_similar_nearby;
    int nb_done_nearby;
} CellMetrics;

typedef struct MEContext {
    const AVClass *class;
    AVMotionEstContext2 me_ctx;

    int cell_size;                      ///< macroblock size
    int search_param;                   ///< search parameter
    int nb_columns, nb_rows, nb_cells;
    int log2_cell_size;

    int frame_number;
    AVFrame *prev, *cur, *next;
    int current_dir;

    CellMetrics *cells_metrics;         ///< Array of size nb_columns * nb_rows for block analysis
    CellMetrics **cells_metrics_sorted_references;
} MEContext;

typedef struct ThreadData {
    AVMotionEstContext2 *me_ctx;
    AVMotionVector *mv_data;
} ThreadData;

#define OFFSET(x) offsetof(MEContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CONST(name, help, val, u) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, 0, 0, FLAGS, .unit = u }

static const AVOption mestimate2_options[] = {
    { "mb_size", "macroblock size", OFFSET(cell_size), AV_OPT_TYPE_INT, {.i64=16}, 8, INT_MAX, FLAGS },
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

    s->log2_cell_size = av_ceil_log2_c(s->cell_size);
    s->cell_size = 1 << s->log2_cell_size;

    s->nb_columns  = inlink->w >> s->log2_cell_size;
    s->nb_rows = inlink->h >> s->log2_cell_size;
    s->nb_cells = s->nb_columns * s->nb_rows;

    if (s->nb_cells == 0)
        return AVERROR(EINVAL);

    s->cells_metrics = av_malloc(sizeof(CellMetrics) * s->nb_cells);
    if (!s->cells_metrics)
        return AVERROR(ENOMEM);

    s->cells_metrics_sorted_references = av_malloc(sizeof(CellMetrics *) * s->nb_cells);
    if (!s->cells_metrics_sorted_references)
        return AVERROR(ENOMEM);

    // Initialize all cells
    for (int i = 0; i < s->nb_cells; i++) {
        CellMetrics *current = &s->cells_metrics[i];
        current->index_y = i / s->nb_columns;
        current->index_x = i % s->nb_columns;
        current->x = current->index_x * s->cell_size;
        current->y = current->index_y * s->cell_size;
        s->cells_metrics_sorted_references[i] = current;

        int index = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0)
                    continue;
                int ny = current->index_y + dy;
                int nx = current->index_x + dx;
                if (nx >= 0 && nx < s->nb_columns &&
                    ny >= 0 && ny < s->nb_rows) {
                    current->neighbours[index++] = &s->cells_metrics[ny * s->nb_columns + nx];
                }
            }
        }
        current->neighbours[index] = NULL;
    }

    ff_me_init_context2(&s->me_ctx, s->cell_size, s->search_param, inlink->w, inlink->h, 0, (s->nb_columns - 1) << s->log2_cell_size, 0, (s->nb_rows - 1) << s->log2_cell_size);
    s->frame_number = 0;

    return 0;
}

static void add_mv_data(AVMotionVector *mv, int cell_size,
                        int x, int y, int x_mv, int y_mv, int dir)
{
    mv->w = cell_size;
    mv->h = cell_size;
    mv->dst_x = x + (cell_size >> 1);
    mv->dst_y = y + (cell_size >> 1);
    mv->src_x = x_mv + (cell_size >> 1);
    mv->src_y = y_mv + (cell_size >> 1);
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

    int slice_size = s->nb_rows / nb_jobs;
    if (slice_size * nb_jobs < s->nb_rows)
        slice_size++;
    int slice_start = jobnr * slice_size;
    int slice_end = min(slice_start + slice_size, s->nb_rows);
    int mv_count = slice_start * s->nb_columns + (dir * s->nb_rows * s->nb_columns);

    for (mb_y = slice_start; mb_y < slice_end; mb_y++) {
        for (mb_x = 0; mb_x < s->nb_columns; mb_x++) {
            do {
                for (mb_y = slice_start; mb_y < slice_end; mb_y++)
                    for (mb_x = 0; mb_x < s->nb_columns; mb_x++) {
                        const int x_mb = (mb_x << s->log2_cell_size);
                        const int y_mb = (mb_y << s->log2_cell_size);
                        int mv[4] = {x_mb, y_mb, 0, 0};
                        ff_me_search_tdls2(me_ctx, x_mb, y_mb, mv);

                        //add_mv_data(td->mv_data + mv_count, s->cell_size, x_mb, y_mb, mv[0], mv[1], 0); // TODO REMOVE

                        CellMetrics* bm = &s->cells_metrics[mv_count++];
                        bm->x_mv = mv[0] - x_mb;
                        bm->y_mv = mv[1] - y_mb;
                        bm->cost = mv[2];
                        bm->similarity = mv[3];
                    }
            } while (0);
        }
    }

    return 0;
}

static void write_debug_file(AVFilterContext *ctx, MEContext *s)
{
    char filename[1024];
    snprintf(filename, sizeof(filename), "Debug/image_%04d.txt", s->frame_number);

    FILE *f = fopen(filename, "w");
    if (!f) {
        av_log(ctx, AV_LOG_ERROR, "Cannot open debug file %s\n", filename);
        return;
    }

    // Write the grid of nb_similar_nearby
    for (int row = 0; row < s->nb_rows; row++) {
        for (int column = 0; column < s->nb_columns; column++) {
            const CellMetrics *bm = &s->cells_metrics[row * s->nb_columns + column];
            // Convert count (0-8) to character
            if (bm->is_done)
                fputc('A' + (8 - bm->nb_similar_nearby), f);
            else
                fputc('a' + (8 - bm->nb_similar_nearby), f);
        }
        fputc('\n', f);
    }

    fclose(f);
}

static inline int get_speed(const CellMetrics *cell) {
    return abs(cell->x_mv) + abs(cell->y_mv);
}

static inline int is_similar_motion(const CellMetrics *cell1, const CellMetrics *cell2, int max_similar_offset) {
    return (abs(cell1->x_mv - cell2->x_mv) <= max_similar_offset) &&
           (abs(cell1->y_mv - cell2->y_mv) <= max_similar_offset);
}

static uint32_t compute_nb_similar_nearby(const CellMetrics *cell, int max_similar_offset) {
    uint32_t similar_count = 0;

    // Iterate through the neighbour array until we hit NULL
    for (int i = 0; cell->neighbours[i] != NULL; i++) {
        if (is_similar_motion(cell, cell->neighbours[i], max_similar_offset)) {
            similar_count++;
        }
    }
    return similar_count;
}

static int finalize_cell(MEContext *s, CellMetrics *cell) {
    cell->is_done = 1;

    // Update done count for neighbors
    for (int i = 0; cell->neighbours[i] != NULL; i++) {
        cell->neighbours[i]->nb_done_nearby++;
    }

    // First, virtually remove all the pixels that's moving from the current cell.
    int nb_pixels_to_move = s->cell_size * s->cell_size;
    cell->nb_next_frame_pixels -= nb_pixels_to_move;

    // Calculate source block grid positions
    int src_block_x = cell->index_x;
    int src_block_y = cell->index_y;

    // Calculate destination position in pixels
    int dst_x = cell->x + cell->x_mv;
    int dst_y = cell->y + cell->y_mv;

    // Calculate which blocks the destination overlaps
    int start_column = dst_x / s->cell_size;
    int start_row = dst_y / s->cell_size;
    int end_column = (dst_x + s->cell_size - 1) / s->cell_size;
    int end_row = (dst_y + s->cell_size - 1) / s->cell_size;

    // Only process the blocks that could be affected
    for (int row = start_row; row <= end_row; row++) {
        for (int column = start_column; column <= end_column; column++) {
            // Bounds checking
            if (column < 0 || column >= s->nb_columns || row < 0 || row >= s->nb_rows)
                continue;

            CellMetrics *target = &s->cells_metrics[row * s->nb_columns + column];

            int overlap_x1 = FFMAX(target->x, dst_x);
            int overlap_y1 = FFMAX(target->y, dst_y);
            int overlap_x2 = FFMIN(target->x + s->cell_size, dst_x + s->cell_size);
            int overlap_y2 = FFMIN(target->y + s->cell_size, dst_y + s->cell_size);

            if (overlap_x1 < overlap_x2 && overlap_y1 < overlap_y2) {
                int overlap_pixels = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1);
                target->nb_next_frame_pixels += overlap_pixels;
                nb_pixels_to_move -= overlap_pixels;
            }
        }
    }

    if (nb_pixels_to_move == 0) {
        int i = 0;
    }
    else {
        int i = 0;
    }

    return 0;
}


// Comparison function for qsort
static int compare_cells_metrics(const void *a, const void *b) {
    const CellMetrics *ref_a = *(const CellMetrics **)a;
    const CellMetrics *ref_b = *(const CellMetrics **)b;

    if (ref_a->is_done != ref_b->is_done) {
        return ref_b->is_done - ref_a->is_done;
    }

    if (ref_b->nb_similar_nearby != ref_a->nb_similar_nearby)
        return ref_b->nb_similar_nearby - ref_a->nb_similar_nearby;



    // First compare by total movement
    int total_mv_a = get_speed(ref_a);
    int total_mv_b = get_speed(ref_b);
    if (total_mv_a != total_mv_b) {
        return total_mv_b - total_mv_a;
    }

    // If equal, compare by similarity_with_neighbourg
    return ref_b->nb_similar_nearby - ref_a->nb_similar_nearby;
}

typedef struct MotionGroup {
    CellMetrics *members[8];  // Maximum 8 neighbors can be in a group
    int count;
    uint32_t cost;
    int nb_pixels_pushed_in;
} MotionGroup;

static int calculate_sweep_overlap(CellMetrics *cell, CellMetrics *moving_cell, int cell_size) {
    int src_x = moving_cell->x;
    int src_y = moving_cell->y;
    int mv_x = moving_cell->x_mv;
    int mv_y = moving_cell->y_mv;

    // We need to calculate intersection with a parallelogram
    // formed by the moving cell's path
    int total_overlap = 0;

    // Break the movement into smaller steps
    int steps = FFMAX(abs(mv_x), abs(mv_y)) / (cell_size/2) + 1;
    float dx = (float)mv_x / steps;
    float dy = (float)mv_y / steps;

    for (int i = 0; i < steps; i++) {
        int curr_x = src_x + (int)(dx * i);
        int curr_y = src_y + (int)(dy * i);

        // Calculate overlap with current position
        int overlap_x1 = FFMAX(cell->x, curr_x);
        int overlap_y1 = FFMAX(cell->y, curr_y);
        int overlap_x2 = FFMIN(cell->x + cell_size, curr_x + cell_size);
        int overlap_y2 = FFMIN(cell->y + cell_size, curr_y + cell_size);

        if (overlap_x1 < overlap_x2 && overlap_y1 < overlap_y2) {
            total_overlap += (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1);
        }
    }

    // Normalize the overlap based on steps to avoid double-counting
    return total_overlap / steps;
}

static int analyze_cell(MEContext *s, CellMetrics *cell, int max_similar_offset) {
    MotionGroup groups[9];  // Maximum possible groups (8 neighbors + cell)
    int group_count = 0;

    // Initialize first group
    groups[0].members[0] = cell;
    groups[0].count = 1;
    groups[0].cost = cell->cost;
    groups[0].nb_pixels_pushed_in = 0;
    group_count = 1;

    // Group neighbors by similar motion vectors
    for (int i = 0; cell->neighbours[i] != NULL; i++) {
        CellMetrics *neighbor = cell->neighbours[i];
        int neighbor_speed = get_speed(neighbor);
        if (neighbor_speed == 0) {
            // Ignore unmoving neighbor
            continue;
        }

        int found_group = 0;

        // Calculate potential overlap for this neighbor
        int overlap_pixels;
        if (get_speed(neighbor) > s->cell_size)
        {
            overlap_pixels = calculate_sweep_overlap(cell, neighbor, s->cell_size);
        }
        else {
            int dst_x = neighbor->x + neighbor->x_mv;
            int dst_y = neighbor->y + neighbor->y_mv;

            int overlap_x1 = FFMAX(cell->x, dst_x);
            int overlap_y1 = FFMAX(cell->y, dst_y);
            int overlap_x2 = FFMIN(cell->x + s->cell_size, dst_x + s->cell_size);
            int overlap_y2 = FFMIN(cell->y + s->cell_size, dst_y + s->cell_size);

            if (overlap_x1 < overlap_x2 && overlap_y1 < overlap_y2) {
                overlap_pixels = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1);
            }
            else {
                overlap_pixels = 0;
            }
        }

        // Try to add to existing group
        for (int g = 0; g < group_count; g++) {
            if (is_similar_motion(groups[g].members[0], neighbor, max_similar_offset)) {
                groups[g].members[groups[g].count++] = neighbor;
                groups[g].nb_pixels_pushed_in += overlap_pixels;
                found_group = 1;
                break;
            }
        }

        // Create new group if needed
        if (!found_group) {
            groups[group_count].members[0] = neighbor;
            groups[group_count].count = 1;
            groups[group_count].cost = -1;
            if (cell->x + neighbor->x_mv >= 0 && cell->x + neighbor->x_mv + s->cell_size < s->me_ctx.x_max
                && cell->y + neighbor->y_mv >= 0 && cell->y + neighbor->y_mv + s->cell_size < s->me_ctx.y_max)
            {
                groups[group_count].cost = ff_me_cmp_sad2(&s->me_ctx,
                                            cell->x,
                                            cell->y,
                                            cell->x + neighbor->x_mv,
                                            cell->y + neighbor->y_mv);
            }
            groups[group_count].nb_pixels_pushed_in = overlap_pixels;
            group_count++;
        }
    }

    // Evaluate each group's motion vector using ff_me_cmp_sad2
    uint32_t best_cost = cell->cost;
    int best_x_mv = cell->x_mv;
    int best_y_mv = cell->y_mv;

    for (int g = 0; g < group_count; g++) {
        if (groups[g].nb_pixels_pushed_in > s->cell_size
            && groups[g].count > 3
             && groups[g].cost - cell->cost < 200
            && get_speed(groups[g].members[0]) > 4) {

            cell->x_mv = groups[g].members[0]->x_mv;
            cell->y_mv = groups[g].members[0]->y_mv;
            cell->cost = groups[g].cost;
            cell->nb_similar_nearby = groups[g].count;
            finalize_cell(s, cell);

            // Recalculate similarity for its neighbors
            for (int i = 0; cell->neighbours[i] != NULL; i++) {
                cell->neighbours[i]->nb_similar_nearby =
                    compute_nb_similar_nearby(cell->neighbours[i], max_similar_offset);
            }
            return 1;
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

    s->frame_number++;

    av_frame_free(&s->prev);
    s->prev = s->cur;
    s->cur  = s->next;
    s->next = frame;

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

    sd = av_frame_new_side_data(out, AV_FRAME_DATA_MOTION_VECTORS, 1 * s->nb_cells * sizeof(AVMotionVector));
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
    me_ctx->search_param = 7;
    me_ctx->data_ref = s->next->data[0]; // prev

    int nb_threads = FFMIN(s->nb_rows, ff_filter_get_nb_threads(ctx));
    // nb_threads = 1; // TODO REMOVE
    ff_filter_execute(ctx, filter_slice, &td, NULL, nb_threads);

    int index_done = 0;
    int min_nb_similar_nearby = 6;
    int min_nb_neighbour_done = 6;
    int min_speed = 8;
    int max_similar_offset = 2;
    int nb_pixels_per_cell = s->cell_size * s->cell_size;
    int too_full_threshold = nb_pixels_per_cell * 3 / 2;
    int too_empty_threshold = nb_pixels_per_cell * 2 / 3;

    // Reset fields and compute the initial value of neighboor_similarity
    for (int i = 0; i < s->nb_cells; i++) {
        CellMetrics *current = &s->cells_metrics[i];
        current->is_done = 0;
        current->nb_next_frame_pixels = nb_pixels_per_cell;
        current->nb_similar_nearby = compute_nb_similar_nearby(current, max_similar_offset);
        current->nb_done_nearby = 0;
    }

    int nb_transformed;
    int nb_finalized;
    do {
        qsort(&s->cells_metrics_sorted_references[index_done], s->nb_cells - index_done, sizeof(CellMetrics *), compare_cells_metrics);

        while(index_done < s->nb_cells && s->cells_metrics_sorted_references[index_done]->is_done) {
            index_done++;
        }

        nb_transformed = 0;
        nb_finalized = 0;
        int maximum_speed_found = 0;
        int maximum_nb_similar_nearby_found = 0;
        for (int i = index_done; i < s->nb_cells; i++) {
            CellMetrics *current = s->cells_metrics_sorted_references[i];
            if (s->frame_number == 3 && current->index_y == 121 && current->index_x == 113) {
                int k = 0;
            }
            maximum_nb_similar_nearby_found = FFMAX(maximum_nb_similar_nearby_found, current->nb_similar_nearby);

            if (current->nb_similar_nearby >= min_nb_similar_nearby && min_speed >= min_speed) {
                maximum_speed_found = FFMAX(maximum_speed_found, get_speed(current));

                finalize_cell(s, current);
                nb_finalized++;
            }
            else if (current->nb_done_nearby >= min_nb_neighbour_done
                 && (current->nb_next_frame_pixels >= too_full_threshold))
            {
                 if (analyze_cell(s, current, max_similar_offset)) {
                     nb_transformed++;
                 }
            }
        }

        if (nb_transformed + nb_finalized == 0) {
            break;
            min_nb_similar_nearby--;
            min_nb_neighbour_done--;
        }
    } while (index_done < s->nb_cells);

    write_debug_file(ctx, s);
    int index = 0;
    for (int i = 0; i < s->nb_cells; i++) {
        CellMetrics *current = &s->cells_metrics[i];
        //if (current->is_done)
            add_mv_data(td.mv_data + index++, s->cell_size, current->x, current->y, current->x - (current->x_mv * 1), current->y - current->y_mv, 0);
    }

    return ff_filter_frame(ctx->outputs[0], out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MEContext *s = ctx->priv;

    av_frame_free(&s->prev);
    av_frame_free(&s->cur);
    av_frame_free(&s->next);
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
