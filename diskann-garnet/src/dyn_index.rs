/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach E: Unlimited-effort paged post-filter.
//!
//! Removes the max_effort cap from the paged filter loop. Instead of stopping
//! after evaluating FILTER-EF candidates, the loop continues until either:
//!   - k matching results are found, OR
//!   - the graph returns 0 candidates (frontier exhausted)
//!
//! This mimics Redis's dual-queue approach where non-matching candidates don't
//! consume the results budget. The graph explores outward from the query point,
//! and the filter is applied purely as a post-processing step.
//!
//! Uses FullPrecision (no BetaFilter) — zero FFI callback overhead during
//! graph traversal. The only FFI calls happen in the post-filter loop.

use crate::{
    BatchFilterCandidateCallback,
    FilterCandidateCallback,
    SearchResults,
    garnet::{Context, GarnetId, Term},
    labels::GarnetQueryLabelProvider,
    provider::{self, GarnetProvider},
};
use diskann::{
    ANNError, ANNResult,
    graph::{InplaceDeleteMethod, SearchOutputBuffer, glue::SearchStrategy, index::{SearchState, SearchStats}, search},
    neighbor::Neighbor,
    provider::{Accessor, BuildQueryComputer, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
use diskann_vector::PreprocessedDistanceFunction;
use std::cmp::{max, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Type-erased version of `DiskANNIndex<GarnetProvider>`.
/// All vector data is passed as untyped byte slices.
pub trait DynIndex: Send + Sync {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn get_attributes(&self, context: &Context, id: &GarnetId) -> Option<Vec<u8>>;

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        batch_filter_callback: BatchFilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element_filtered(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        batch_filter_callback: BatchFilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    fn approximate_count(&self) -> u64;

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()>;

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool;

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool;
}

impl<T: VectorRepr> DynIndex for DiskANNIndex<GarnetProvider<T>> {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.insert(
            FullPrecision,
            context,
            id,
            bytemuck::cast_slice::<u8, T>(data),
        )
    }

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .set_attributes(context, id, data)
            .map_err(|e| e.into())
    }

    fn get_attributes(&self, context: &Context, id: &GarnetId) -> Option<Vec<u8>> {
        self.inner.provider().get_attributes(context, id)
    }

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        if let Some((labels, beta)) = filter {
            let beta_filter = BetaFilter::new(FullPrecision, Arc::new(labels.clone()), beta);
            self.search(&beta_filter, context, query, params, output)
        } else {
            self.search(&FullPrecision, context, query, params, output)
        }
    }

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _, GarnetId>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = rt.block_on(accessor.get_element(iid))?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector(context, data_bytes, params, filter, output)
    }

    /// Filtered search dispatcher.
    ///
    /// Two modes selected via a high-bit flag in max_effort:
    ///   - Normal (max_effort < PAGED_FLAG): two-queue filtered search with convergence
    ///   - Paged (max_effort >= PAGED_FLAG): paged unlimited post-filter loop
    ///
    /// The paged mode uses DiskANN's native paged search (search_internal + next_search_results)
    /// which explores the graph in beam-width pages and post-filters each page.
    ///
    /// To select paged mode from C#, set FILTER-EF to (desired_effort | 0x40000000).
    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        _label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        batch_filter_callback: BatchFilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        const PAGED_FLAG: usize = 0x40000000; // bit 30
        const BATCH_MASK: usize = 0x00FF0000; // bits 16-23

        let query = bytemuck::cast_slice::<u8, T>(data);
        let k = params.k_value().get();
        let ef = params.l_value().get();

        // Extract batch_size from bits 16-23 of max_effort
        let batch_size = (max_effort & BATCH_MASK) >> 16;
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        if max_effort & PAGED_FLAG != 0 {
            // Paged unlimited mode: use the real EF as beam width.
            // ef may be inflated by C#'s Math.Max(EF, FilterEF|PAGED_FLAG|BATCH_BITS),
            // so strip the flag and batch bits to get a reasonable beam width.
            let l_value = ef & 0x0000FFFF;

            let mut state = self.start_paged_search(
                FullPrecision,
                context,
                query,
                l_value,
            )?;

            paged_filter_loop_unlimited(
                self,
                context,
                &mut state,
                k,
                l_value,
                filter_callback,
                batch_filter_callback,
                batch_size,
                output,
                0,
            )
        } else {
            // Two-queue mode (default)
            // Strip batch_size bits and paged flag from max_effort to get the actual effort cap
            let raw_effort = max_effort & 0x0000FFFF;
            let effort_cap = if raw_effort > 0 { raw_effort } else { 100_000 };

            two_queue_filtered_search(
                self.inner.provider(),
                context,
                query,
                k,
                ef,
                filter_callback,
                batch_filter_callback,
                batch_size,
                effort_cap,
                output,
            )
        }
    }

    fn search_element_filtered(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        batch_filter_callback: BatchFilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _, GarnetId>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = rt.block_on(accessor.get_element(iid))?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector_filtered(
            context,
            data_bytes,
            params,
            label_filter,
            filter_callback,
            batch_filter_callback,
            max_effort,
            output,
        )
    }

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()> {
        self.inplace_delete(
            FullPrecision,
            context,
            id,
            3,
            InplaceDeleteMethod::TwoHopAndOneHop,
        )
    }

    fn approximate_count(&self) -> u64 {
        self.inner.provider().max_internal_id() as u64
    }

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .maybe_set_start_point(context, bytemuck::cast_slice::<u8, T>(data))
            .map_err(|e| e.into())
    }

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool {
        self.inner.provider().vector_iid_exists(context, id)
    }

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool {
        self.inner.provider().vector_id_exists(context, id)
    }
}

/// Paged search loop WITHOUT max_effort cap.
///
/// Continues fetching pages of candidates from the graph until either:
/// - k matching results are collected, or
/// - the graph frontier is exhausted (next_search_results returns 0)
///
/// This allows the search to explore deeply into the graph for low-selectivity
/// queries, where many candidates need to be evaluated to find matches.
fn paged_filter_loop_unlimited<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    filter_callback: FilterCandidateCallback,
    batch_filter_callback: BatchFilterCandidateCallback,
    batch_size: usize,
    output: &mut SearchResults<'_>,
    _max_effort: usize,
) -> ANNResult<SearchStats>
where
    T: VectorRepr,
    S: SearchStrategy<GarnetProvider<T>, [T]>,
{
    let mut batch = vec![Neighbor::default(); l_value];
    let mut total_candidates = 0usize;
    let mut total_evaluated = 0usize;
    let mut total_passed = 0usize;
    let mut pages = 0u32;
    let mut graph_ns = 0u64;
    let mut filter_ns = 0u64;
    let mut extid_ns = 0u64;
    let total_start = std::time::Instant::now();
    let mut early_exit = false;

    loop {

        let page_start = std::time::Instant::now();
        let count = index.next_search_results(context, state, l_value, &mut batch)?;
        graph_ns += page_start.elapsed().as_nanos() as u64;
        total_candidates += count;
        pages += 1;

        if count == 0 {
            break;
        }

        for candidates_chunk in batch[..count].chunks(batch_size) {
            let cb_start = std::time::Instant::now();

            if batch_size > 1 {
                if let Some(bcb) = batch_filter_callback {
                    let ids: Vec<u32> = candidates_chunk.iter().map(|c| c.id).collect();
                    let mut pass_buf = vec![0u8; candidates_chunk.len()];
                    unsafe { bcb(context.0, ids.as_ptr(), candidates_chunk.len() as u32, pass_buf.as_mut_ptr()); }
                    filter_ns += cb_start.elapsed().as_nanos() as u64;

                    for (i, candidate) in candidates_chunk.iter().enumerate() {
                        total_evaluated += 1;
                        if pass_buf[i] == 0 { continue; }
                        total_passed += 1;

                        let eid_start = std::time::Instant::now();
                        let eid_result = index.inner.provider().to_external_id(context, candidate.id);
                        extid_ns += eid_start.elapsed().as_nanos() as u64;

                        if let Ok(eid) = eid_result {
                            if output.push(eid, candidate.distance).is_full() || output.current_len() >= k {
                                early_exit = true;
                                break;
                            }
                        }
                    }
                    if early_exit { break; }
                    continue;
                }
            }

            // Single callback fallback
            for candidate in candidates_chunk {
                total_evaluated += 1;
                let f_start = std::time::Instant::now();
                let passes = unsafe { filter_callback(context.0, candidate.id) != 0 };
                filter_ns += f_start.elapsed().as_nanos() as u64;

                if !passes { continue; }
                total_passed += 1;

                let eid_start = std::time::Instant::now();
                let eid_result = index.inner.provider().to_external_id(context, candidate.id);
                extid_ns += eid_start.elapsed().as_nanos() as u64;

                if let Ok(eid) = eid_result {
                    if output.push(eid, candidate.distance).is_full() || output.current_len() >= k {
                        early_exit = true;
                        break;
                    }
                }
            }
            if early_exit { break; }
        }

        if early_exit {
            break;
        }
    }

    let total_us = total_start.elapsed().as_micros();
    let stats = SearchStats { cmps: state.scratch.cmps, hops: state.scratch.hops, result_count: output.current_len() as u32, range_search_second_round: false };
    eprintln!(
        "[paged-unlimited] l={l_value} batch_size={batch_size} pages={pages} candidates={total_candidates} evaluated={total_evaluated} passed={total_passed} found={} cmps={} hops={} early_exit={early_exit} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        stats.result_count, stats.cmps, stats.hops, graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    Ok(stats)
}

/// Check if bit is set in a u64 bitvec. Returns true if was already set.
#[inline(always)]
fn bitvec_test_and_set(bits: &mut [u64], id: u32) -> bool {
    let word = (id >> 6) as usize;
    let bit = 1u64 << (id & 63);
    if word >= bits.len() { return true; } // out of bounds → treat as visited
    let was_set = bits[word] & bit != 0;
    bits[word] |= bit;
    was_set
}

/// Two-queue filtered search with two-phase convergence.
///
/// Phase 1: Standard beam search with convergence detection. Explores the graph
/// using distance-based convergence (like HNSW). Filter is checked inline.
/// If k filtered results are found, convergence triggers quickly.
///
/// Phase 2 (fallback): If phase 1 converges without k results (low selectivity),
/// continues expanding from remaining candidates without convergence check,
/// until k results are found or effort cap is hit.
///
/// max_candidates controls the total exploration budget (FilterEF from C#).
fn two_queue_filtered_search<T: VectorRepr>(
    provider: &GarnetProvider<T>,
    context: &Context,
    query: &[T],
    k: usize,
    ef: usize,
    filter_callback: FilterCandidateCallback,
    batch_filter_callback: BatchFilterCandidateCallback,
    batch_size: usize,
    max_candidates: usize,
    output: &mut SearchResults<'_>,
) -> ANNResult<SearchStats> {
    let total_start = std::time::Instant::now();
    let mut graph_ns = 0u64;
    let mut filter_ns = 0u64;
    let mut extid_ns = 0u64;

    let mut accessor = provider::FullAccessor::new(provider, context, true);
    let computer = accessor.build_query_computer(query)?;

    // Beam convergence: use a fixed beam size for graph traversal quality.
    // ef from C# = Max(EF, FilterEF) — can be inflated by FilterEF.
    // The beam should use the base graph EF (typically 200), not FilterEF.
    // Cap beam at 1000 to prevent FilterEF from inflating it needlessly.
    let explore_ef = ef.min(1000).max(200);

    // Use a bitvec for visited instead of HashSet — O(1) with better cache locality
    let max_id = provider.max_internal_id() as usize;
    let visited_words = (max_id + 64) / 64;
    let mut visited_bits = vec![0u64; visited_words];
    let mut visited_count: usize = 0;

    // Min-heap for candidates (closest first)
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(explore_ef);
    // Max-heap tracking ef best distances seen (for phase 1 convergence)
    let mut best_seen: BinaryHeap<OrderedF32> = BinaryHeap::with_capacity(explore_ef + 1);
    // Results: max-heap capped at result_cap (worst-first for pruning) — truncated to k at end
    let result_cap = (k * 5).max(k);
    let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(result_cap + 1);

    let mut cmps: u32 = 0;
    let mut hops: u32 = 0;
    let mut evaluated: u32 = 0;
    let mut passed: u32 = 0;
    let mut phase1_converged = false;
    let mut final_converged = false;

    // Seed with medoid (id=0)
    let start_id: u32 = 0;
    bitvec_test_and_set(&mut visited_bits, start_id);
    visited_count += 1;

    let start_dist = if let Some(cached) = provider.start_point_cache.get(&start_id) {
        let d = computer.evaluate_similarity(&*cached);
        cmps += 1;
        d
    } else {
        let read_ids = vec![4u32, start_id];
        let mut d = f32::MAX;
        provider.callbacks().read_multi_lpiid(
            context.term(Term::Vector),
            &read_ids,
            |_i, v: &[T]| {
                d = computer.evaluate_similarity(v);
            },
        );
        cmps += 1;
        d
    };

    candidates.push(Reverse((OrderedF32(start_dist), start_id)));
    best_seen.push(OrderedF32(start_dist));

    // Check filter on start node
    evaluated += 1;
    let cb_start = std::time::Instant::now();
    if unsafe { filter_callback(context.0, start_id) != 0 } {
        passed += 1;
        results.push((OrderedF32(start_dist), start_id));
    }
    filter_ns += cb_start.elapsed().as_nanos() as u64;

    // Pre-allocated buffers for neighbor expansion (avoid per-hop allocation)
    let mut batch_ids: Vec<u32> = Vec::with_capacity(128);
    let mut pending: Vec<(u32, f32)> = Vec::with_capacity(64);

    // ─── Phase 1: Beam-converged exploration with inline filtering ───
    while !candidates.is_empty() {
        // Cap on filter evaluations, not distance comparisons.
        // Graph exploration (beam convergence) runs freely — only the expensive
        // FFI filter callback count is bounded by max_candidates (FilterEF).
        if evaluated as usize >= max_candidates {
            break;
        }

        let Reverse((OrderedF32(cur_dist), current)) = candidates.pop().unwrap();
        hops += 1;

        // Convergence: beam search has settled
        if best_seen.len() >= explore_ef {
            let ef_threshold = best_seen.peek().unwrap().0;
            if cur_dist > ef_threshold {
                phase1_converged = true;
                if results.len() >= result_cap {
                    final_converged = true;
                    break;
                }
                break;
            }
        }

        // Result-based convergence
        if results.len() >= result_cap {
            let worst_result = results.peek().unwrap().0 .0;
            if cur_dist > worst_result {
                phase1_converged = true;
                final_converged = true;
                break;
            }
        }

        // --- Expand neighbors of `current` ---
        {
            let g_start = std::time::Instant::now();
            accessor.get_neighbors_internal(current, None);
            graph_ns += g_start.elapsed().as_nanos() as u64;

            batch_ids.clear();
            pending.clear();

            for &nid in accessor.id_buffer.iter() {
                if !bitvec_test_and_set(&mut visited_bits, nid) {
                    visited_count += 1;
                    if nid == 0 {
                        if let Some(cached) = provider.start_point_cache.get(&nid) {
                            let dist = computer.evaluate_similarity(&*cached);
                            cmps += 1;
                            let beam_threshold = if best_seen.len() >= explore_ef { best_seen.peek().unwrap().0 } else { f32::MAX };
                            if dist < beam_threshold || best_seen.len() < explore_ef {
                                candidates.push(Reverse((OrderedF32(dist), nid)));
                                best_seen.push(OrderedF32(dist));
                                if best_seen.len() > explore_ef { best_seen.pop(); }
                            }
                            pending.push((nid, dist));
                        }
                    } else {
                        batch_ids.push(4);
                        batch_ids.push(nid);
                    }
                }
            }

            if !batch_ids.is_empty() {
                let g2_start = std::time::Instant::now();
                provider.callbacks().read_multi_lpiid(
                    context.term(Term::Vector),
                    &batch_ids,
                    |i, v: &[T]| {
                        let nid = batch_ids[i as usize * 2 + 1];
                        let dist = computer.evaluate_similarity(v);
                        cmps += 1;

                        let beam_threshold = if best_seen.len() >= explore_ef { best_seen.peek().unwrap().0 } else { f32::MAX };
                        if dist < beam_threshold || best_seen.len() < explore_ef {
                            candidates.push(Reverse((OrderedF32(dist), nid)));
                            best_seen.push(OrderedF32(dist));
                            if best_seen.len() > explore_ef { best_seen.pop(); }
                        }

                        pending.push((nid, dist));
                    },
                );
                graph_ns += g2_start.elapsed().as_nanos() as u64;
            }

            // Filter checks — MUST be outside read_multi_lpiid (no nested FFI)
            let f_start = std::time::Instant::now();
            filter_candidates_batch(
                context.0, &pending, filter_callback, batch_filter_callback, batch_size,
                &mut results, result_cap, &mut evaluated, &mut passed,
            );
            filter_ns += f_start.elapsed().as_nanos() as u64;
        }
    }

    // ─── Phase 2: Multi-hop batched exploration without beam convergence ───
    if phase1_converged && !final_converged && results.len() < k {
        const MULTI_HOP_BATCH: usize = 16;
        let mut hop_batch: Vec<u32> = Vec::with_capacity(MULTI_HOP_BATCH);

        'phase2: loop {
            if evaluated as usize >= max_candidates || candidates.is_empty() {
                break;
            }

            hop_batch.clear();
            while hop_batch.len() < MULTI_HOP_BATCH {
                if candidates.is_empty() { break; }
                let Reverse((OrderedF32(cur_dist), current)) = candidates.pop().unwrap();
                hops += 1;

                if results.len() >= k {
                    let worst_result = results.peek().unwrap().0 .0;
                    if cur_dist > worst_result {
                        final_converged = true;
                        break 'phase2;
                    }
                }

                hop_batch.push(current);
            }

            batch_ids.clear();
            pending.clear();

            for &current in &hop_batch {
                let g_start = std::time::Instant::now();
                accessor.get_neighbors_internal(current, None);
                graph_ns += g_start.elapsed().as_nanos() as u64;

                for &nid in accessor.id_buffer.iter() {
                    if !bitvec_test_and_set(&mut visited_bits, nid) {
                        visited_count += 1;
                        if nid == 0 {
                            if let Some(cached) = provider.start_point_cache.get(&nid) {
                                let dist = computer.evaluate_similarity(&*cached);
                                cmps += 1;
                                candidates.push(Reverse((OrderedF32(dist), nid)));
                                pending.push((nid, dist));
                            }
                        } else {
                            batch_ids.push(4);
                            batch_ids.push(nid);
                        }
                    }
                }
            }

            if !batch_ids.is_empty() {
                let g2_start = std::time::Instant::now();
                provider.callbacks().read_multi_lpiid(
                    context.term(Term::Vector),
                    &batch_ids,
                    |i, v: &[T]| {
                        let nid = batch_ids[i as usize * 2 + 1];
                        let dist = computer.evaluate_similarity(v);
                        cmps += 1;

                        candidates.push(Reverse((OrderedF32(dist), nid)));
                        pending.push((nid, dist));
                    },
                );
                graph_ns += g2_start.elapsed().as_nanos() as u64;
            }

            let f_start = std::time::Instant::now();
            filter_candidates_batch(
                context.0, &pending, filter_callback, batch_filter_callback, batch_size,
                &mut results, result_cap, &mut evaluated, &mut passed,
            );
            filter_ns += f_start.elapsed().as_nanos() as u64;
        }
    }

    // Collect results sorted by distance
    let mut sorted_results: Vec<(f32, u32)> = results
        .into_vec()
        .into_iter()
        .map(|(OrderedF32(d), id)| (d, id))
        .collect();
    sorted_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let found = sorted_results.len().min(k);
    let eid_start = std::time::Instant::now();
    for &(dist, id) in sorted_results.iter().take(k) {
        if let Ok(eid) = provider.to_external_id(context, id) {
            let _ = output.push(eid, dist);
        }
    }
    extid_ns += eid_start.elapsed().as_nanos() as u64;

    let total_us = total_start.elapsed().as_micros();
    let stats = SearchStats {
        cmps,
        hops,
        result_count: output.current_len() as u32,
        range_search_second_round: false,
    };
    eprintln!(
        "[two-queue] ef={ef} k={k} rcap={result_cap} max_effort={max_candidates} batch_size={batch_size} hops={hops} cmps={cmps} evaluated={evaluated} \
         passed={passed} found={found} converged={final_converged} ph1_conv={phase1_converged} \
         total={total_us}µs graph={}µs filter={}µs extid={}µs visited={}",
        graph_ns / 1000, filter_ns / 1000, extid_ns / 1000, visited_count
    );
    Ok(stats)
}

/// Evaluate filter for a batch of candidates.
/// Uses batch_filter_callback if available and batch_size > 1,
/// otherwise falls back to single filter_callback.
#[inline]
fn filter_candidates_batch(
    context_id: u64,
    pending: &[(u32, f32)],
    filter_callback: FilterCandidateCallback,
    batch_cb: BatchFilterCandidateCallback,
    batch_size: usize,
    results: &mut BinaryHeap<(OrderedF32, u32)>,
    result_cap: usize,
    evaluated: &mut u32,
    passed: &mut u32,
) {
    if batch_size > 1 {
        if let Some(bcb) = batch_cb {
            for chunk in pending.chunks(batch_size) {
                let ids: Vec<u32> = chunk.iter().map(|&(nid, _)| nid).collect();
                let mut pass_buf = vec![0u8; chunk.len()];
                unsafe { bcb(context_id, ids.as_ptr(), chunk.len() as u32, pass_buf.as_mut_ptr()); }
                for (i, &(nid, dist)) in chunk.iter().enumerate() {
                    if results.len() >= result_cap && dist > results.peek().unwrap().0 .0 {
                        continue;
                    }
                    *evaluated += 1;
                    if pass_buf[i] != 0 {
                        *passed += 1;
                        results.push((OrderedF32(dist), nid));
                        if results.len() > result_cap { results.pop(); }
                    }
                }
            }
            return;
        }
    }

    // Single callback fallback
    for &(nid, dist) in pending {
        if results.len() >= result_cap && dist > results.peek().unwrap().0 .0 {
            continue;
        }
        *evaluated += 1;
        if unsafe { filter_callback(context_id, nid) != 0 } {
            *passed += 1;
            results.push((OrderedF32(dist), nid));
            if results.len() > result_cap { results.pop(); }
        }
    }
}

/// Wrapper for f32 that implements Ord (needed for BinaryHeap).
/// NaN is treated as greater than everything (pushed to the end).
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            // NaN handling: NaN > everything
            if self.0.is_nan() && other.0.is_nan() {
                std::cmp::Ordering::Equal
            } else if self.0.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        })
    }
}
