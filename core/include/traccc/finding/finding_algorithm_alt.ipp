/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/finding/candidate_link.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/projections.hpp"

// System include
#include <algorithm>
//#include <iostream>

namespace traccc {

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::host
finding_algorithm<stepper_t, navigator_t>::operator()(
    const detector_type& det, const bfield_type& field,
    const measurement_collection_types::host& measurements,
    const bound_track_parameters_collection_types::host& seeds) const {

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    // Check contiguity of the measurements
    assert(
        host::is_contiguous_on(measurement_module_projection(), measurements));

    // Get index ranges in the measurement container per detector surface
    std::vector<unsigned int> meas_ranges;
    meas_ranges.reserve(det.surfaces().size());

    for (const auto& sf_desc : det.surfaces()) {
        // Measurements can only be found on sensitive surfaces
        if (!sf_desc.is_sensitive()) {
            // Lower range index is the upper index of the previous range
            // This is guaranteed by the measurement sorting step
            const auto sf_idx{sf_desc.index()};
            const unsigned int lo{sf_idx == 0u ? 0u : meas_ranges[sf_idx - 1u]};

            // Hand the upper index of the previous range through to assign
            // the lower index of the next valid range correctly
            meas_ranges.push_back(lo);
            continue;
        }

        auto up = std::upper_bound(measurements.begin(), measurements.end(),
                                   sf_desc.barcode(), measurement_bcd_comp());
        meas_ranges.push_back(
            static_cast<unsigned int>(std::distance(measurements.begin(), up)));
    }

    /*****************************************************************
     * CKF Preparations
     *****************************************************************/
    const std::size_t n_seeds{seeds.size()};
    const unsigned int n_max_prop_streams{
        math::min(m_cfg.max_num_branches_per_seed, 20000u)};
    const unsigned int n_max_branches_per_surface{
        math::min(m_cfg.max_num_branches_per_surface, 10u)};

    // Measurement trace per track
    std::vector<trace_state> track_traces;
    track_traces.resize(n_seeds * n_max_prop_streams);

    // Compatible measurements and filtered track params on a given surface
    std::vector<candidate> candidates{};
    candidates.reserve(n_max_branches_per_surface);

    // Create propagator
    propagator_type propagator(m_cfg.propagation);

    // Propagation states: One inner vector per seed, which contains
    // the branched propagation states for the respective seed
    std::vector<std::vector<typename propagator_type::state>> prop_states;
    prop_states.reserve(n_seeds);

    // Create initial propagation flow for each seed
    for (const auto& seed : seeds) {
        auto& prop_states_per_seed = prop_states.emplace_back();
        prop_states_per_seed.reserve(n_max_prop_streams);

        auto& propagation = prop_states_per_seed.emplace_back(seed, field, det);

        propagation.set_particle(
            detail::correct_particle_hypothesis(m_cfg.ptc_hypothesis, seed));
        propagation._navigation.set_volume(seed.surface_link().volume());

        // Create actor states
        typename aborter::state s0{m_cfg.propagation.stepping.path_limit};
        typename transporter::state s1;
        typename interactor::state s2;
        typename resetter::state s3;

        auto actor_states = detray::tie(s0, s1, s2, s3);

        propagator.init(propagation, actor_states);

        // Make sure the CKF can start on a sensitive surface
        if (propagation._heartbeat &&
            !propagation._navigation.is_on_sensitive()) {
            propagator.propagate_to_next(propagation, actor_states);
        }
        // Either exited detector by portal right away or are on first
        // sensitive surface
        assert(propagation._navigation.is_complete() ||
               propagation._navigation.is_on_sensitive());
    }

    // Step through the sensitive surfaces along the tracks
    const auto n_steps{static_cast<int>(m_cfg.max_track_candidates_per_track)};
    for (int step = 0; step < n_steps; step++) {
        // Step through all track candidates (branches) for a given seed
        for (unsigned int seed_idx = 0u; seed_idx < n_seeds; seed_idx++) {

            auto& prop_streams = prop_states[seed_idx];
            const auto n_tracks{static_cast<unsigned int>(prop_streams.size())};
            for (unsigned int trk_idx = 0u; trk_idx < n_tracks; ++trk_idx) {
                // The propagation flow for this track
                auto& propagation = prop_streams[trk_idx];

                // Propagation is no longer alive (track finished or hit error)
                if (!propagation._heartbeat) {
                    continue;
                }
                assert(propagation._navigation.is_on_sensitive());

                // Conditions context for this propagation
                const cxt_t ctx{};
                // Get current detector surface (sensitive)
                const auto sf = propagation._navigation.get_surface();

                /***************************************************************
                 * Find compatible measurements
                 ***************************************************************/

                // Iterate over the measurements for this surface
                const auto sf_idx{sf.index()};
                const unsigned int lo{sf_idx == 0u ? 0u
                                                   : meas_ranges[sf_idx - 1]};
                const unsigned int up{meas_ranges[sf_idx]};

                for (unsigned int meas_id = lo; meas_id < up; meas_id++) {

                    track_state<algebra_type> trk_state(measurements[meas_id]);

                    // Run the Kalman update on a copy of the track parameters
                    bound_track_parameters bound_param(
                        propagation._stepping._bound_params);
                    const bool res = sf.template visit_mask<
                        gain_matrix_updater<algebra_type>>(trk_state,
                                                           bound_param);

                    // Found a good measurement?
                    const auto chi2 = trk_state.filtered_chi2();
                    if (res && chi2 < m_cfg.chi2_max) {
                        candidates.emplace_back(trk_state.filtered(), meas_id,
                                                chi2);
                    }
                }

                /***************************************************************
                 * Update current propagation flow and branch
                 ***************************************************************/

                /// Update the bound and free track parameters after the KF
                auto update_params =
                    [&ctx, &sf](typename propagator_type::state& prop_state,
                                const bound_track_parameters& new_params) {
                        auto& current_params =
                            prop_state._stepping._bound_params;
                        current_params = new_params;
                        prop_state._stepping() =
                            sf.bound_to_free_vector(ctx, current_params);
                    };

                const unsigned int trace_idx{seed_idx * n_max_prop_streams +
                                             trk_idx};

                // Count hole in case no measurements were found
                if (candidates.empty()) {
                    track_traces[trace_idx].n_skipped++;

                    // If number of skips is larger than the maximal value,
                    // consider the track to be finished
                    if (track_traces[trace_idx].n_skipped >
                        m_cfg.max_num_skipping_per_cand) {
                        propagation._heartbeat &=
                            propagation._navigation.abort();
                        assert(!propagation._heartbeat);
                    }
                } else {
                    // Consider only the best candidates
                    std::sort(candidates.begin(), candidates.end());

                    // Update the track parameters in the current propagation
                    // with the closest candidate - use the rest for branching
                    update_params(propagation, candidates[0u].filtered_params);

                    // Number of potential new branches
                    unsigned int n_branches = math::min(
                        static_cast<unsigned int>(candidates.size()) - 1u,
                        n_max_branches_per_surface);

                    // Number of allowed new branches
                    auto allowed_branches{
                        static_cast<int>(n_max_prop_streams) -
                        static_cast<int>(prop_streams.size())};
                    allowed_branches =
                        math::signbit(allowed_branches) ? 0 : allowed_branches;

                    // Create new branches
                    n_branches =
                        math::min(n_branches,
                                  static_cast<unsigned int>(allowed_branches));
                    for (unsigned int i = 0u; i < n_branches; ++i) {
                        // Clone current propagation flow for new branch
                        prop_streams.push_back(propagation);
                        update_params(prop_streams.back(),
                                      candidates[i + 1].filtered_params);

                        // Copy the measurements that were gathered up until now
                        // to the trace of the new branch
                        // @TODO: Need multitrajectory?
                        const std::size_t branch_idx{prop_streams.size() - 1u};
                        const std::size_t new_trace_idx{
                            seed_idx * n_max_prop_streams + branch_idx};
                        track_traces[new_trace_idx].meas_trace =
                            track_traces[trace_idx].meas_trace;

                        // Add new measurement
                        track_traces[new_trace_idx].meas_trace.push_back(
                            measurements[candidates[i + 1].meas_id]);
                    }

                    // Next measurement for original propagation flow
                    track_traces[trace_idx].meas_trace.push_back(
                        measurements[candidates[0u].meas_id]);
                }

                candidates.clear();
            }

            /*******************************************************
             * Propagate all tracks of this seed to the next surface
             *******************************************************/
            for (auto& propagation : prop_streams) {
                // Propagation is no longer alive (track finished or hit error)
                if (!propagation._heartbeat) {
                    continue;
                }
                assert(propagation._navigation.is_on_sensitive());

                // Create actor states
                // TODO: This does not work, the actor states need to be kept
                // alive as well
                typename aborter::state s0{
                    m_cfg.propagation.stepping.path_limit};
                typename transporter::state s1;
                typename interactor::state s2;
                typename resetter::state s3;

                // Propagate to the next surface
                propagator.propagate_to_next(propagation,
                                             detray::tie(s0, s1, s2, s3));
            }
        }
    }

    /**********************
     * Build tracks
     **********************/
    track_candidate_container_types::host output_candidates;
    output_candidates.reserve(2u * n_seeds);

    // Step through all track candidates for a given seed
    for (unsigned int seed_idx = 0u; seed_idx < n_seeds; seed_idx++) {

        for (unsigned int trk_idx = 0u; trk_idx < prop_states[seed_idx].size();
             ++trk_idx) {
            const unsigned int trace_idx{seed_idx * n_max_prop_streams +
                                         trk_idx};

            assert(prop_states[seed_idx][trk_idx]._stepping._path_length > 0.f);
            assert(track_traces[trace_idx].meas_trace.size() > 0);

            if (track_traces[trace_idx].meas_trace.size() >=
                m_cfg.min_track_candidates_per_track) {
                output_candidates.push_back(
                    bound_track_parameters{seeds[seed_idx]},
                    std::move(track_traces[trace_idx].meas_trace));
            }
        }
    }

    return output_candidates;
}

}  // namespace traccc
