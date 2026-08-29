#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The catalogue of every generated asset, and the run that produces it.

:data:`_FIGURE_FUNCS` is the order ``make graphs`` walks and the list
``--figure`` resolves against; :data:`_ANIMATIONS` is the same for the clips.
The registries are also where the run is scheduled from: which figures must
render their four variants in one worker to reuse a memoised analysis, how
expensive each task is so the heavy ones are submitted first, and the
name-consistency assertions that make a rename fail at import instead of
silently degrouping a cached figure or dropping a clip's field builder.
:func:`main` is the command line over the same tables.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .aircraft import (
    generate_aircraft_atmospheric_absorption,
    generate_airport_contour,
    generate_airport_noise,
    generate_airport_segment_breakdown,
    generate_airport_segment_corrections,
    generate_airport_sor,
    generate_anp_contour,
    generate_anp_npd,
    generate_anp_procedural_profile,
    generate_anp_profile,
    generate_epnl,
    generate_rotorcraft_contour,
    generate_rotorcraft_flight_conditions,
    generate_rotorcraft_flyover_event,
    generate_rotorcraft_ground_effect,
    generate_rotorcraft_hemisphere,
    generate_rotorcraft_hover_ring,
    generate_rotorcraft_insertion_loss,
    generate_rotorcraft_kinematics,
    generate_rotorcraft_mean_ground_plane,
    generate_rotorcraft_terrain_screening,
)
from .building import (
    generate_background_correction_regimes,
    generate_ceiling_plenum_flanking,
    generate_composite_facade_weak_element,
    generate_dbhr_global_index,
    generate_extended_insulation_rating,
    generate_facade_elevation_geometry,
    generate_facade_field_insulation,
    generate_facade_prediction,
    generate_fast_reverberation_correction,
    generate_field_airborne_insulation,
    generate_flanking_level_difference,
    generate_flanking_transmission,
    generate_heavy_impact_sources,
    generate_impact_rating,
    generate_insulation_adaptation_terms,
    generate_insulation_rating,
    generate_insulation_uncertainty_demo,
    generate_intensity_element_insulation,
    generate_intensity_field_indicator,
    generate_intensity_insulation,
    generate_lab_insulation_result,
    generate_lab_versus_field_insulation,
    generate_radiated_power_outdoor,
    generate_survey_impact_insulation,
    generate_survey_insulation,
)
from .building_design import (
    generate_aperture_slit_geometry,
    generate_coupling_term_regimes,
    generate_detailed_impact_paths,
    generate_detailed_prediction_paths,
    generate_double_wall_geometry,
    generate_floating_floor_prediction,
    generate_floor_covering_improvement,
    generate_impact_prediction_terms,
    generate_installed_structure_borne,
    generate_masonry_wall_ties,
    generate_orthotropic_transmission_loss,
    generate_panel_insulation_concept,
    generate_plateau_transmission_loss,
    generate_prediction_flanking_demo,
    generate_radiation_efficiency_panels,
    generate_single_panel_rating,
    generate_soft_covering_prediction,
    generate_structure_borne_conversion,
    generate_structure_borne_power,
    generate_tapping_force_spectrum,
)
from .correlation_analysis import (
    generate_cepstrum_echo,
    generate_cepstrum_variants,
    generate_correlation_normalizations,
    generate_envelope_spectrum,
    generate_gcc_phat_delay,
    generate_hilbert_envelope,
    generate_ir_alignment,
    generate_lifter_split,
    generate_synchronous_average,
    generate_tsa_noise_reduction,
)
from .devices import (
    generate_channel_weight_map,
    generate_distortion,
    generate_duct_attenuation_elements,
    generate_duct_mode_cut_on,
    generate_duct_path_cascade,
    generate_duct_regenerated_noise,
    generate_duct_sheet_verification,
    generate_enclosure_insertion_loss,
    generate_enclosure_required_tl,
    generate_expansion_chamber_geometry,
    generate_extended_tube_geometry,
    generate_fan_sound_power,
    generate_feedback_stability,
    generate_field_indicators,
    generate_frequency_response,
    generate_helmholtz_branch_geometry,
    generate_hvac_elbow_flow_noise,
    generate_hvac_end_reflection,
    generate_intensity_class,
    generate_intensity_demo,
    generate_intensity_scan_power,
    generate_intermodulation_tests,
    generate_itu_r_468_weighting,
    generate_k1_k2_corrections,
    generate_k_weighting_response,
    generate_loudness_gating,
    generate_loudness_range,
    generate_loudspeaker_directivity,
    generate_loudspeaker_impedance,
    generate_loudspeaker_response,
    generate_loudspeaker_thd,
    generate_microphone_directivity,
    generate_microphone_distortion,
    generate_microphone_noise,
    generate_microphone_noise_weightings,
    generate_microphone_patterns,
    generate_microphone_positions_hemisphere,
    generate_microphone_response,
    generate_modulation_distortion,
    generate_partial_power_map,
    generate_phase_decomposition,
    generate_piston_baffle_geometry,
    generate_piston_directivity,
    generate_piston_radiation_impedance,
    generate_plenum_geometry,
    generate_pp_probe_geometry,
    generate_precision_anechoic_power,
    generate_precision_positions_arrays,
    generate_program_loudness,
    generate_quarter_wave_geometry,
    generate_quasi_peak_meter,
    generate_radiation_efficiency,
    generate_radiation_plate_geometry,
    generate_reverberation_correction_terms,
    generate_room_to_room_chain,
    generate_room_to_room_partitions,
    generate_silencer_chain_geometry,
    generate_silencer_expansion_chamber,
    generate_silencer_extended_tube,
    generate_silencer_insertion_loss,
    generate_silencer_selection,
    generate_silencer_side_branch,
    generate_sound_power_grades_declaration,
    generate_sound_power_intensity_result,
    generate_sound_power_pressure_result,
    generate_sound_power_reverberation_result,
    generate_sound_reinforcement_geometry,
    generate_spacer_bandwidth,
    generate_swept_sine_harmonic_responses,
    generate_swept_sine_methods,
    generate_swept_sine_thd,
    generate_true_peak_intersample,
    generate_vibration_sound_power,
)
from .environment import (
    generate_air_absorption_alpha,
    generate_atmospheric_attenuation,
    generate_atmospheric_pe_range,
    generate_atmospheric_ray_fan,
    generate_atmospheric_refraction,
    generate_atmospheric_sound_speed_profiles,
    generate_barrier_geometry,
    generate_barrier_insertion_loss_methods,
    generate_barrier_thickness_gain,
    generate_cnossos_rail_components,
    generate_cnossos_rail_directivity,
    generate_cnossos_rail_emission,
    generate_cnossos_rail_roughness_shift,
    generate_cnossos_road_corrections,
    generate_cnossos_road_emission,
    generate_cnossos_road_gradient,
    generate_cnossos_road_speed_law,
    generate_cnossos_road_surfaces,
    generate_ground_effect_spherical,
    generate_ground_reflection_coefficient,
    generate_impulse_prominence,
    generate_impulsive_sound_onsets,
    generate_iso9613_screening_anatomy,
    generate_lden_profile,
    generate_outdoor_attenuation_breakdown,
    generate_outdoor_level_cascade,
    generate_rd1367_activity_assessment,
    generate_rd1367_kf_ki,
    generate_rd1367_tonal_correction,
    generate_rd1367_vs_iso_tonal,
    generate_refraction_homogeneous_check,
    generate_shadow_zone_map,
    generate_tonal_audibility,
    generate_wind_turbine_apparent_power,
    generate_wind_turbine_audibility_criterion,
    generate_wind_turbine_tonality,
)
from .fields import (
    _absorption_placement_fields,
    _aperture_fields,
    _barrier_fields,
    _coincidence_fields,
    _diffusion_fields,
    _dispersion_fields,
    _duct_cut_on_fields,
    _ducting_fields,
    _expansion_chamber_fields,
    _ground_effect_fields,
    _halfspace_fields,
    _impedance_tube_fields,
    _metadiffuser_fields,
    _mode_conversion_fields,
    _pillar_fields,
    _plate_junction_fields,
    _poster_ss_for,
    _radiation_efficiency_fields,
    _refraction_fields,
    _room_mode_fields,
    _seabed_fields,
    _side_branch_fields,
    _slit_absorber_fields,
    _transmission_tube_fields,
    animate_elastic_coincidence,
    animate_elastic_halfspace_waves,
    animate_elastic_mode_conversion,
    animate_elastic_plate_junction,
    animate_elastic_radiation_efficiency,
    animate_fdtd_absorption_placement,
    animate_fdtd_aperture_slit,
    animate_fdtd_barrier,
    animate_fdtd_critical_angle,
    animate_fdtd_diffusion,
    animate_fdtd_dispersion,
    animate_fdtd_duct_cut_on,
    animate_fdtd_ducting,
    animate_fdtd_expansion_chamber,
    animate_fdtd_ground_effect,
    animate_fdtd_impedance_tube,
    animate_fdtd_metadiffuser,
    animate_fdtd_pillar_hall,
    animate_fdtd_refraction,
    animate_fdtd_room_modes,
    animate_fdtd_side_branch,
    animate_fdtd_slit_absorber,
    animate_fdtd_transmission_tube,
    animate_standing_wave_tube,
)
from .i18n import set_lang
from .io import generate_signal_provenance
from .materials import (
    generate_absorber_stack_geometry,
    generate_absorption_rating,
    generate_absorption_uncertainty,
    generate_adrienne_window,
    generate_airflow_resistance,
    generate_biot_frame_resonance,
    generate_biot_waves,
    generate_critical_coupling_impedance,
    generate_diffuse_field_absorption,
    generate_diffuser_modulation,
    generate_diffuser_prediction,
    generate_diffusion_goniometer_geometry,
    generate_diffusion_measurement_chain,
    generate_diffusion_polar,
    generate_dynamic_stiffness,
    generate_dynamic_stiffness_rig_geometry,
    generate_effective_kappa,
    generate_enclosed_gas_stiffness,
    generate_floating_floor_transmissibility,
    generate_flow_resistivity_window,
    generate_graded_slit_absorber,
    generate_helmholtz_resonator_geometry,
    generate_impedance_tube,
    generate_impedance_tube_geometry,
    generate_impedance_tube_result,
    generate_insitu_absorption,
    generate_insitu_method_windows,
    generate_insitu_setup_geometry,
    generate_limp_frame_effective_density,
    generate_metadiffuser_absorption,
    generate_metadiffuser_geometry,
    generate_metadiffuser_phase_match,
    generate_metadiffuser_polar,
    generate_metadiffuser_spectrum,
    generate_mpp_absorption_peak,
    generate_oblique_absorption,
    generate_porous_absorber_designs,
    generate_porous_medium_model,
    generate_porous_model_comparison,
    generate_qrd_geometry,
    generate_qrd_working_band,
    generate_scattering_coefficient,
    generate_sheet_transfer_impedance,
    generate_slit_absorber_geometry,
    generate_slow_sound_absorber,
    generate_slow_sound_dispersion,
    generate_sound_absorption_inversion,
    generate_sound_absorption_measurement,
    generate_standing_wave_envelope,
    generate_transfer_matrix_tl,
    generate_transmission_tube_geometry,
    generate_tube_working_ranges,
)
from .media import _extract_poster
from .metrology import (
    generate_calibration_narrowband_bias,
    generate_calibration_stability,
    generate_dbfs_versus_spl,
    generate_rice_level_crossings,
    generate_rice_nongaussian_screen,
    generate_rice_peak_distribution,
    generate_runs_test,
    generate_stationarity_glide_blind_spot,
    generate_stationarity_test,
    generate_trend_test,
    generate_uncertainty,
    generate_uncertainty_correlation,
    generate_uncertainty_gum_vs_mc,
)
from .perception import (
    generate_age_threshold_fractiles,
    generate_age_threshold_sex_and_spread,
    generate_annoyance_weightings,
    generate_equal_loudness_contours,
    generate_erb_bandwidth,
    generate_exposure_budget,
    generate_exposure_uncertainty,
    generate_fluctuation_strength,
    generate_fluctuation_strength_specific,
    generate_hearing_threshold,
    generate_hms_modulation_bandpass,
    generate_htlan_compression,
    generate_loudness_models_comparison,
    generate_loudness_pattern,
    generate_moore_glasberg_specific_loudness,
    generate_moore_glasberg_time_loudness,
    generate_nipts_audiogram,
    generate_nipts_level_growth,
    generate_noise_induced_hearing_loss,
    generate_psychoacoustic_annoyance,
    generate_sharpness_pair_and_targets,
    generate_sharpness_weighting,
    generate_sii_band_procedures,
    generate_sii_hearing_loss,
    generate_sii_masking_chain,
    generate_sii_octave_masking_blindness,
    generate_sii_vocal_efforts,
    generate_sottek_specific_fluctuation,
    generate_sottek_specific_loudness,
    generate_sottek_specific_roughness,
    generate_sottek_specific_tonality,
    generate_speech_intelligibility,
    generate_standard_speech_spectrum,
    generate_sti_band_mti,
    generate_sti_curve,
    generate_sti_level_dependence,
    generate_sti_mtf_curves,
    generate_stoi_band_scores,
    generate_stoi_intelligibility,
    generate_stoi_segment_scores,
    generate_tnr_pr_comparison,
    generate_tonality_roughness_demo,
    generate_tonality_spectrum,
    generate_tone_audibility,
    generate_tone_audibility_levels,
    generate_tone_audibility_uncertainty,
    generate_tone_prominence_assessment,
    generate_two_tone_separation,
    generate_zwicker_time_varying,
)
from .room import (
    generate_absorption_per_table,
    generate_decay_range_bias,
    generate_decay_signatures,
    generate_deconvolution_snr_gain,
    generate_enclosed_space_absorption,
    generate_enclosed_space_air_term,
    generate_enclosed_space_objects,
    generate_excitation_robustness,
    generate_excitation_signals,
    generate_image_source_anisotropy,
    generate_image_source_bands,
    generate_image_source_order_convergence,
    generate_image_source_plan,
    generate_image_source_reflectogram,
    generate_impulse_response,
    generate_modal_count_per_band,
    generate_nc_blind_spot,
    generate_open_plan_decay,
    generate_open_plan_line_geometry,
    generate_open_plan_quality,
    generate_rectangular_room_modes,
    generate_restaurant_crowd_noise,
    generate_reverberation_model_absorption,
    generate_reverberation_models,
    generate_room_noise_criteria,
    generate_room_parameters_bands,
    generate_room_proportion_modes,
    generate_schroeder_decay,
    generate_source_distance_bias,
    generate_steady_state_directivity,
    generate_steady_state_field,
    generate_sweep_distortion_separation,
)
from .schematics import (
    animate_block_vs_exponential,
    animate_comb_filtering,
    animate_dynamic_stiffness_sweep,
    animate_epnl_flyover,
    animate_feedback_howl,
    animate_flanking_paths,
    animate_image_source_buildup,
    animate_instantaneous_intensity,
    animate_intensity_scan_power,
    animate_iso717_shift,
    animate_loudness_gating,
    animate_modulation_transfer,
    animate_onset_detection,
    animate_power_two_rooms,
    animate_schroeder,
    animate_specific_loudness,
    animate_sweep_deconvolution,
    animate_time_weighting_ballistics,
)
from .signals import (
    generate_architecture_tradeoff,
    generate_ballistics_vs_duration,
    generate_block_processing_continuity,
    generate_c_minus_a_spectrum,
    generate_class_mask_architectures,
    generate_class_mask_overlay,
    generate_crossover_plot,
    generate_decomposition_plot,
    generate_dose_exchange,
    generate_energy_vs_arithmetic_mean,
    generate_filter_class0_mask,
    generate_filter_responses,
    generate_filter_type_comparison,
    generate_g_weighting_response,
    generate_group_delay_comparison,
    generate_leakage_floor,
    generate_level_distribution,
    generate_ln_levels_example,
    generate_multichannel_response,
    generate_parametric_eq_cascade,
    generate_parametric_eq_family,
    generate_peak_oversampling,
    generate_pole_migration,
    generate_sel_concept,
    generate_signal_responses,
    generate_slm_level_track,
    generate_slm_third_octave,
    generate_special_weighting_responses,
    generate_spectrogram_example,
    generate_streaming_level_seams,
    generate_survey_channel_average,
    generate_time_weighting_plot,
    generate_tone_burst_iec,
    generate_weighting_accuracy_hf,
    generate_weighting_class_mask,
    generate_weighting_responses,
    generate_zero_phase_comparison,
)
from .simulation import (
    generate_elastic_halfspace_waves,
    generate_elastic_probe_traces,
    generate_fdtd_domain_geometry,
    generate_fdtd_plane_wave_launch,
    generate_fdtd_room_modes,
    generate_fdtd_simulation,
    generate_metadiffuser_meshed_panel,
    generate_metadiffuser_ntff_polar,
    generate_scholte_interface_wave,
)
from .spectral_estimation import (
    generate_calibrated_spectrogram,
    generate_coherent_output_snr,
    generate_cross_spectral_density_delay,
    generate_miso_coherence,
    generate_multitaper_psd_confidence,
    generate_noise_colors,
    generate_psd_confidence_smoothing,
    generate_psd_segment_tradeoff,
    generate_window_functions_tradeoff,
    generate_zoom_fft_resolution,
)
from .system_measurement import (
    generate_golay_ir,
    generate_regularized_inversion,
    generate_resampling_antialias,
    generate_shaped_sweep,
    generate_tone_burst_train,
)
from .theme import set_theme
from .underwater import (
    generate_detection_range,
    generate_eigenray_arrivals,
    generate_gaussian_beam_caustic,
    generate_marine_mammal_assessment,
    generate_marine_mammal_audiograms,
    generate_marine_mammal_exposure_functions,
    generate_marine_mammal_weighting,
    generate_normal_modes,
    generate_numerical_propagation,
    generate_ocean_ambient_noise,
    generate_pe_paraxial_error,
    generate_pile_driving,
    generate_piling_campaign_accumulation,
    generate_ray_turning_point,
    generate_seabed_reflection,
    generate_seabed_reflection_coefficient,
    generate_seawater_absorption,
    generate_ship_source_level,
    generate_ship_traffic_noise,
    generate_sonar_budget,
    generate_sonar_equation,
    generate_sound_speed_models,
    generate_underwater_propagation_loss,
    generate_underwater_sound_speed,
    generate_weston_regimes,
)
from .vibration import (
    generate_bearing_fault_envelope,
    generate_daily_vibration_exposure,
    generate_envelope_chain_steps,
    generate_experimental_sea_clf,
    generate_hav_vwf_lifetime,
    generate_infinite_mobilities,
    generate_junction_kij_thickness,
    generate_junction_plate_geometry,
    generate_junction_transmission,
    generate_machine_fault_families,
    generate_mechanical_mobility,
    generate_mobility_random_error,
    generate_mobility_result_lines,
    generate_multiple_shock,
    generate_rigid_mass_calibration,
    generate_shock_dose_measures,
    generate_spinal_response_peaks,
    generate_transfer_stiffness,
    generate_vibration_weighting,
    generate_vibration_weighting_family,
    generate_weighted_acceleration,
)

_FIGURE_FUNCS: tuple[Callable[[str], None], ...] = (
    generate_filter_type_comparison,
    generate_filter_responses,
    generate_signal_responses,
    generate_multichannel_response,
    generate_decomposition_plot,
    generate_weighting_responses,
    generate_special_weighting_responses,
    generate_g_weighting_response,
    generate_equal_loudness_contours,
    generate_time_weighting_plot,
    generate_crossover_plot,
    generate_parametric_eq_family,
    # Feature documentation plots (levels, spectrogram, zero-phase, weighting accuracy)
    generate_spectrogram_example,
    generate_ln_levels_example,
    generate_zero_phase_comparison,
    generate_weighting_accuracy_hf,
    # Docs-enrichment plots (group delay, IEC toneburst, block continuity, class mask)
    generate_group_delay_comparison,
    generate_tone_burst_iec,
    generate_block_processing_continuity,
    generate_class_mask_overlay,
    generate_filter_class0_mask,
    generate_weighting_class_mask,
    generate_calibration_stability,
    generate_calibration_narrowband_bias,
    generate_dbfs_versus_spl,
    # The audio-files guide: the calibrated waveform a measurement WAV comes
    # back as, next to the bext provenance card that arrived with it.
    generate_signal_provenance,
    generate_sel_concept,
    generate_ballistics_vs_duration,
    # Filter banks: the z-plane behind the multirate design, and the cascade
    # the parametric-EQ snippet builds.
    generate_pole_migration,
    generate_parametric_eq_cascade,
    generate_architecture_tradeoff,
    generate_class_mask_architectures,
    generate_leakage_floor,
    generate_streaming_level_seams,
    generate_survey_channel_average,
    generate_c_minus_a_spectrum,
    # Integrated and statistical levels: the distribution the percentiles are
    # defined on, the energy-mean rule, the inter-sample peak and the dose.
    generate_level_distribution,
    generate_energy_vs_arithmetic_mean,
    generate_peak_oversampling,
    generate_dose_exchange,
    # The sound level meter walkthrough: the readouts it computes, drawn.
    generate_slm_level_track,
    generate_slm_third_octave,
    generate_lden_profile,
    generate_rd1367_activity_assessment,
    generate_dbhr_global_index,
    generate_tonality_spectrum,
    # Psychoacoustics / intensity plots (loudness, STI, p-p intensity)
    generate_loudness_pattern,
    generate_zwicker_time_varying,
    generate_tone_audibility_uncertainty,
    generate_two_tone_separation,
    generate_tnr_pr_comparison,
    generate_sharpness_pair_and_targets,
    generate_sottek_specific_roughness,
    generate_sottek_specific_fluctuation,
    generate_sti_curve,
    generate_intensity_demo,
    # Room / building acoustics plots (ISO 18233 excitations + IR, Schroeder
    # decay, ISO 717-1/-2 ratings)
    generate_excitation_signals,
    generate_impulse_response,
    generate_schroeder_decay,
    # What the ISO 18233 acquisition buys and what it costs: the effective
    # SNR of a deconvolved sweep, the harmonic packets at negative arrival
    # times, the bias of a microphone inside d_min, the modal count an octave
    # band averages over, and the sweep's tolerance of time variance.
    generate_deconvolution_snr_gain,
    generate_sweep_distortion_separation,
    generate_source_distance_bias,
    generate_modal_count_per_band,
    generate_excitation_robustness,
    generate_insulation_rating,
    generate_impact_rating,
    # Building-acoustics prediction / uncertainty (EN 12354-1, ISO 12999-1)
    generate_prediction_flanking_demo,
    generate_facade_prediction,
    generate_intensity_insulation,
    generate_survey_insulation,
    generate_floor_covering_improvement,
    generate_heavy_impact_sources,
    generate_ceiling_plenum_flanking,
    generate_insulation_adaptation_terms,
    generate_background_correction_regimes,
    generate_fast_reverberation_correction,
    generate_lab_versus_field_insulation,
    generate_composite_facade_weak_element,
    generate_intensity_field_indicator,
    generate_masonry_wall_ties,
    generate_floating_floor_prediction,
    generate_soft_covering_prediction,
    generate_flanking_transmission,
    generate_reverberation_models,
    # Rooms/prediction: model behaviour against the mean absorption,
    # the EN 12354-6 air and object terms, the image-source order
    # horizon, anisotropy and bands, room proportion, the steady-state
    # directivity pair, the decay signatures and the decay-range bias.
    generate_reverberation_model_absorption,
    generate_enclosed_space_air_term,
    generate_enclosed_space_objects,
    generate_image_source_order_convergence,
    generate_image_source_anisotropy,
    generate_image_source_bands,
    generate_room_proportion_modes,
    generate_steady_state_directivity,
    generate_decay_signatures,
    generate_decay_range_bias,
    generate_dynamic_stiffness,
    generate_floating_floor_transmissibility,
    generate_enclosed_gas_stiffness,
    generate_mechanical_mobility,
    generate_junction_transmission,
    generate_bearing_fault_envelope,
    # Machine diagnostics (Norton & Karczub 8.4): the gear, motor and fan
    # families as patterns, and the three steps of the envelope route.
    generate_machine_fault_families,
    generate_envelope_chain_steps,
    generate_experimental_sea_clf,
    generate_plateau_transmission_loss,
    generate_orthotropic_transmission_loss,
    generate_transfer_stiffness,
    generate_rigid_mass_calibration,
    generate_vibration_sound_power,
    generate_structure_borne_power,
    generate_installed_structure_borne,
    generate_structure_borne_conversion,
    generate_coupling_term_regimes,
    generate_tapping_force_spectrum,
    generate_detailed_impact_paths,
    generate_radiation_efficiency_panels,
    generate_tone_audibility,
    generate_absorption_uncertainty,
    generate_insulation_uncertainty_demo,
    # Outdoor propagation & occupational exposure (ISO 9613-1/2, ISO 9612)
    generate_air_absorption_alpha,
    generate_atmospheric_attenuation,
    generate_outdoor_attenuation_breakdown,
    generate_cnossos_road_emission,
    generate_cnossos_road_speed_law,
    generate_cnossos_road_corrections,
    generate_cnossos_road_gradient,
    generate_cnossos_road_surfaces,
    generate_ground_effect_spherical,
    generate_ground_reflection_coefficient,
    generate_barrier_thickness_gain,
    generate_iso9613_screening_anatomy,
    generate_outdoor_level_cascade,
    generate_atmospheric_refraction,
    generate_shadow_zone_map,
    generate_refraction_homogeneous_check,
    generate_wind_turbine_apparent_power,
    generate_wind_turbine_audibility_criterion,
    generate_rd1367_tonal_correction,
    generate_rd1367_kf_ki,
    generate_rd1367_vs_iso_tonal,
    # CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3).
    generate_cnossos_rail_emission,
    generate_cnossos_rail_roughness_shift,
    generate_cnossos_rail_components,
    generate_cnossos_rail_directivity,
    generate_exposure_uncertainty,
    generate_exposure_budget,
    # Materials: absorption rating, airflow resistance, impedance tube
    # (ISO 11654, ISO 9053-1/-2, ISO 10534-1/-2, ASTM E2611)
    generate_absorption_rating,
    generate_airflow_resistance,
    generate_impedance_tube,
    # Porous materials & multilayer absorbers (Mechel / Bies / Cox & D'Antonio)
    generate_porous_absorber_designs,
    generate_limp_frame_effective_density,
    generate_biot_frame_resonance,
    generate_absorber_stack_geometry,
    # Slow-sound slit + Helmholtz-resonator perfect absorbers (Jimenez et al.)
    generate_metadiffuser_geometry,
    generate_metadiffuser_polar,
    generate_metadiffuser_absorption,
    generate_metadiffuser_phase_match,
    generate_metadiffuser_spectrum,
    generate_metadiffuser_ntff_polar,
    generate_slow_sound_absorber,
    generate_slow_sound_dispersion,
    generate_critical_coupling_impedance,
    generate_graded_slit_absorber,
    generate_slit_absorber_geometry,
    generate_helmholtz_resonator_geometry,
    # Scattering/diffusion, in-situ road absorption, precision sound power
    # (ISO 17497-1/-2, ISO 13472-1, ISO 3745 / ISO 9614-3)
    generate_scattering_coefficient,
    generate_diffusion_measurement_chain,
    generate_diffusion_polar,
    generate_diffuser_prediction,
    generate_diffuser_modulation,
    generate_qrd_working_band,
    generate_qrd_geometry,
    generate_impedance_tube_geometry,
    generate_tube_working_ranges,
    generate_standing_wave_envelope,
    generate_transmission_tube_geometry,
    generate_insitu_absorption,
    generate_adrienne_window,
    generate_insitu_method_windows,
    generate_precision_anechoic_power,
    generate_intensity_scan_power,
    # Sound power result spectra for the three most-used routes
    # (ISO 3744 enveloping surface, ISO 3741 reverberation room,
    # ISO 9614-2 intensity scanning)
    generate_sound_power_pressure_result,
    generate_sound_power_reverberation_result,
    generate_sound_power_intensity_result,
    # Human vibration (ISO 8041-1, ISO 2631-1/-2/-4, ISO 5349-1/-2,
    # Directive 2002/44/EC): frequency weighting, weighted a_w, daily A(8)
    generate_vibration_weighting,
    generate_vibration_weighting_family,
    generate_weighted_acceleration,
    generate_shock_dose_measures,
    generate_daily_vibration_exposure,
    generate_hav_vwf_lifetime,
    # Speech intelligibility (ANSI S3.5-1997): band audibility and the SII.
    generate_speech_intelligibility,
    generate_sii_vocal_efforts,
    generate_standard_speech_spectrum,
    generate_sii_band_procedures,
    generate_sii_masking_chain,
    generate_sii_octave_masking_blindness,
    generate_impulse_prominence,
    # Room-noise criteria (ANSI S12.2-2019): NC tangency and RC Mark II,
    # and the pair of rooms the NC number cannot tell apart.
    generate_room_noise_criteria,
    generate_nc_blind_spot,
    # Hearing threshold (ISO 7029 age-related, ISO 389-7 reference).
    generate_hearing_threshold,
    generate_age_threshold_sex_and_spread,
    # Noise-induced hearing loss (ISO 1999 NIPTS and HTLAN).
    generate_noise_induced_hearing_loss,
    generate_nipts_level_growth,
    generate_htlan_compression,
    # Multiple-shock whole-body vibration (ISO 2631-5 Clause 5 + Annex C).
    generate_tonal_audibility,
    generate_multiple_shock,
    generate_spinal_response_peaks,
    # Sound absorption in enclosed spaces (EN 12354-6 Clause 4).
    generate_enclosed_space_absorption,
    # Measurement uncertainty (GUM Guide 98-3 + Supplement 1 Monte Carlo).
    generate_uncertainty,
    generate_uncertainty_gum_vs_mc,
    generate_uncertainty_correlation,
    generate_stationarity_glide_blind_spot,
    generate_rice_nongaussian_screen,
    # Psychoacoustics / open-plan plots (sharpness weighting, spatial decay)
    generate_sharpness_weighting,
    generate_open_plan_decay,
    # The same four quantities read against the two ends of ISO 3382-3
    # Annex A, and the Long Eq. (17.53)-(17.54) absorption-per-table window.
    generate_open_plan_quality,
    generate_absorption_per_table,
    # Rectangular room modes (Long Ch. 8), restaurant crowd self-noise
    # (Long Ch. 17) and the ERB_N / Cam auditory-filter scale.
    generate_rectangular_room_modes,
    generate_restaurant_crowd_noise,
    generate_erb_bandwidth,
    # Advanced psychoacoustics: ECMA-418-2 Sottek model and Moore-Glasberg
    # ISO 532-2/-3 loudness (models, specific loudness, sound quality
    # metrics, time-varying loudness).
    generate_loudness_models_comparison,
    generate_sottek_specific_loudness,
    generate_tonality_roughness_demo,
    # ECMA-418-2 fluctuation strength (Clause 9) vs roughness (Clause 7):
    # the complementary slow/fast modulation band-passes.
    generate_hms_modulation_bandpass,
    generate_moore_glasberg_time_loudness,
    # Fluctuation strength (Fastl & Zwicker Eq. 10.2 + Osses 2016 signal model)
    # and psychoacoustic annoyance (Fastl & Zwicker Eqs 16.2-16.4).
    generate_fluctuation_strength,
    generate_psychoacoustic_annoyance,
    generate_annoyance_weightings,
    # Electroacoustics: distortion metrics (IEC 60268-3) and
    # frequency-response / coherence estimators (Bendat & Piersol).
    generate_distortion,
    generate_frequency_response,
    # The ITU-R BS.468-4 network behind the weighted THD, the AES17 CCIR-RMS
    # noise figures and the dB(468) microphone self-noise, and the products
    # each of the three scalar intermodulation tests counts.
    generate_itu_r_468_weighting,
    generate_intermodulation_tests,
    # Swept-sine harmonic separation: THD(f) by order from one synchronized
    # sweep (Farina 2000 / Novak et al. 2015), the separated harmonic
    # responses themselves, and the synchronized/Farina band and phase.
    generate_swept_sine_thd,
    generate_swept_sine_harmonic_responses,
    generate_swept_sine_methods,
    # Long's gain-before-feedback structure, one open microphone and four.
    generate_feedback_stability,
    # The first-order microphone family and the two self-noise weightings.
    generate_microphone_patterns,
    generate_microphone_noise_weightings,
    # Far-field directivity (beam) pattern of a baffled circular piston.
    generate_piston_directivity,
    # Single-concept rated-characteristic .plot() figures, sharing their panel
    # drawing with the .report() fiches (IEC 60268-5 loudspeaker, IEC 60268-4
    # microphone).
    generate_loudspeaker_response,
    generate_loudspeaker_impedance,
    generate_loudspeaker_thd,
    generate_loudspeaker_directivity,
    generate_microphone_response,
    generate_microphone_directivity,
    generate_microphone_noise,
    generate_microphone_distortion,
    # Calibrated spectral analysis: PSD with chi-square confidence interval
    # and 1/3-octave smoothing on exact-slope pink noise (Bendat & Piersol).
    generate_psd_confidence_smoothing,
    generate_psd_segment_tradeoff,
    generate_noise_colors,
    # Thomson multitaper density of a short record with its chi-square
    # band against the single-taper estimate (Percival & Walden 1993).
    generate_multitaper_psd_confidence,
    # Time-frequency analysis: calibrated STFT spectrogram in dB SPL and
    # the zoom FFT resolving sub-bin tone pairs (Bendat & Piersol).
    generate_calibrated_spectrogram,
    generate_zoom_fft_resolution,
    # Signal toolbox: IEC 60268-1 tone bursts (single and repetitive train)
    # and the window figures of merit (Harris 1978).
    generate_tone_burst_train,
    generate_window_functions_tradeoff,
    # Broadcast: programme loudness and true peak (ITU-R BS.1770-5 /
    # EBU R 128 with Tech 3341/3342).
    generate_program_loudness,
    generate_k_weighting_response,
    # Broadcast: the ITU-R BS.468-4 psophometric quasi-peak meter reading
    # the clause 2.2 burst train.
    generate_quasi_peak_meter,
    # Correlation / time-delay estimation: GCC-PHAT vs the direct
    # correlator on a colored signal pair (Knapp & Carter 1976).
    generate_gcc_phat_delay,
    # Cepstral analysis: echo detection on the power cepstrum (Havelock
    # Chs. 27/87) and the envelope spectrum of an AM tone (B&P 13.3).
    generate_cepstrum_echo,
    generate_envelope_spectrum,
    # Time synchronous averaging of a periodic waveform in noise (McFadden 1987).
    generate_synchronous_average,
    # Multiple-input/output coherence (Bendat & Piersol Ch. 7).
    generate_miso_coherence,
    # Data qualification: reverse arrangement trend and stationarity tests
    # level-crossing / peak statistics (Bendat & Piersol 10.3 / 5.5).
    generate_trend_test,
    generate_stationarity_test,
    generate_rice_level_crossings,
    generate_rice_nongaussian_screen,
    generate_rice_peak_distribution,
    # Regularized inversion (Kirkeby) and the Mueller-Massarani shaped sweep.
    generate_regularized_inversion,
    generate_shaped_sweep,
    # Objective intelligibility: STOI vs ESTOI over SNR for stationary and
    # modulated maskers (Taal et al. 2011 / Jensen & Taal 2016).
    generate_stoi_intelligibility,
    # Room acoustics: the synthetic image-source room impulse response as a
    # reflectogram of mirror-image reflections by order (Kuttruff 4.1).
    generate_image_source_reflectogram,
    generate_image_source_plan,
    generate_open_plan_line_geometry,
    generate_sound_reinforcement_geometry,
    # Underwater acoustics: ship radiated noise / monopole source
    # level (ISO 17208) and pile-driving sound exposure (ISO 18406).
    generate_ship_source_level,
    generate_pile_driving,
    # Aircraft noise: ICAO Annex 16 Effective Perceived Noise Level.
    generate_epnl,
    # Wind-turbine noise: IEC 61400-11 tonal audibility.
    generate_wind_turbine_tonality,
    # Underwater propagation: propagation loss, sound-speed profile and the
    # sonar equation.
    generate_underwater_propagation_loss,
    generate_weston_regimes,
    generate_underwater_sound_speed,
    generate_sonar_equation,
    # Underwater fauna: regulatory auditory weighting (NMFS 2024).
    generate_marine_mammal_weighting,
    # Underwater propagation: seabed reflection, ambient noise (Wenz) and
    # ship-traffic source level (JOMOPANS-ECHO).
    generate_seabed_reflection,
    generate_seabed_reflection_coefficient,
    generate_ocean_ambient_noise,
    generate_ship_traffic_noise,
    # Underwater propagation: numerical solvers (modes/rays/PE).
    generate_numerical_propagation,
    generate_normal_modes,
    generate_ray_turning_point,
    generate_eigenray_arrivals,
    generate_gaussian_beam_caustic,
    generate_pe_paraxial_error,
    # Underwater propagation: the model choices and the budget they feed.
    generate_seawater_absorption,
    generate_sound_speed_models,
    generate_detection_range,
    generate_sonar_budget,
    # Underwater fauna: audiograms, exposure functions and the assessment.
    generate_marine_mammal_audiograms,
    generate_marine_mammal_exposure_functions,
    generate_marine_mammal_assessment,
    generate_piling_campaign_accumulation,
    # Aircraft atmospheric absorption: SAE ARP 5534 band method.
    generate_aircraft_atmospheric_absorption,
    # Airport noise: ECAC Doc 29 noise-power-distance curves, the per-segment
    # corrections and the single event they assemble into.
    generate_airport_noise,
    generate_airport_contour,
    generate_airport_sor,
    generate_airport_segment_breakdown,
    generate_airport_segment_corrections,
    # The EASA ANP fleet database that feeds that chain for a real aircraft.
    generate_anp_npd,
    generate_anp_profile,
    generate_anp_procedural_profile,
    generate_anp_contour,
    # Rotorcraft: the ECAC Doc 32 hemisphere source and its propagation.
    generate_rotorcraft_hemisphere,
    generate_rotorcraft_hover_ring,
    generate_rotorcraft_ground_effect,
    generate_rotorcraft_flyover_event,
    generate_rotorcraft_contour,
    generate_rotorcraft_flight_conditions,
    generate_rotorcraft_kinematics,
    generate_rotorcraft_mean_ground_plane,
    generate_rotorcraft_insertion_loss,
    generate_rotorcraft_terrain_screening,
    # 2D FDTD wave simulation (public API concept figure), the one-way
    # plane-wave launcher measured on its own scene, and the meshed
    # metadiffuser panel the transfer matrix homogenises.
    generate_fdtd_simulation,
    generate_fdtd_plane_wave_launch,
    generate_metadiffuser_meshed_panel,
    # Elastic P-SV FDTD: half-space snapshot with P/S/Rayleigh fronts,
    # and the water-column probe history the result plots by default.
    generate_elastic_halfspace_waves,
    generate_elastic_probe_traces,
    # Elastic FDTD fluid-solid coupling: the Scholte interface wave.
    generate_scholte_interface_wave,
    # Theoretical panel sound insulation (single/double wall, radiation, slit).
    generate_panel_insulation_concept,
    generate_silencer_expansion_chamber,
    generate_expansion_chamber_geometry,
    # Rooms & materials result plots: ISO 354 measurement, ISO 10534-2 tube
    # result, ASTM E2611 transfer matrix, porous models, MPP peak, Paris
    # integral, Bies steady-state field, ISO 3382 per-band parameters and the
    # rigid-box FDTD mode oracle.
    generate_sound_absorption_measurement,
    generate_sound_absorption_inversion,
    generate_impedance_tube_result,
    generate_transfer_matrix_tl,
    generate_porous_medium_model,
    generate_porous_model_comparison,
    generate_mpp_absorption_peak,
    generate_sheet_transfer_impedance,
    generate_diffuse_field_absorption,
    generate_oblique_absorption,
    generate_biot_waves,
    generate_effective_kappa,
    generate_flow_resistivity_window,
    generate_steady_state_field,
    generate_room_parameters_bands,
    generate_fdtd_room_modes,
    # Building & structure-borne result figures (guide figure coverage):
    # ISO 717 enlarged range, ISO 16283 field chains, ISO 10052 survey impact,
    # ISO 10140 laboratory quantities, ISO 15186-1 small elements, ISO 10848
    # flanking descriptors, EN 12354-2/-4 predictions, the rated Sharp panel,
    # the wave-approach junction Kij and the ISO 7626 mobility lines.
    generate_extended_insulation_rating,
    generate_field_airborne_insulation,
    generate_facade_field_insulation,
    generate_survey_impact_insulation,
    generate_lab_insulation_result,
    generate_intensity_element_insulation,
    generate_flanking_level_difference,
    generate_impact_prediction_terms,
    generate_detailed_prediction_paths,
    generate_radiated_power_outdoor,
    generate_single_panel_rating,
    generate_junction_kij_thickness,
    generate_mobility_result_lines,
    # ISO 7626: the reference mobilities of infinite structures and the
    # Annex A averaging cost of the random-error criterion.
    generate_infinite_mobilities,
    generate_mobility_random_error,
    # Perception, hearing and speech: single-concept result figures drawn by
    # the results' own .plot() (ECMA-418-1 tone prominence, ISO/PAS 20065 tone
    # levels, ISO/PAS 1996-3 onsets, ISO 532-2 and ECMA-418-2 patterns, the
    # Osses fluctuation strength, ANSI S3.5 audibility with hearing loss,
    # ISO 7029 / ISO 1999 thresholds, STOI band scores and the STI MTI bars).
    generate_tone_prominence_assessment,
    generate_tone_audibility_levels,
    generate_impulsive_sound_onsets,
    generate_moore_glasberg_specific_loudness,
    generate_sottek_specific_tonality,
    generate_fluctuation_strength_specific,
    generate_sii_hearing_loss,
    generate_age_threshold_fractiles,
    generate_nipts_audiogram,
    generate_stoi_band_scores,
    generate_stoi_segment_scores,
    generate_sti_band_mti,
    generate_sti_mtf_curves,
    generate_sti_level_dependence,
    # Atmospheric refraction (profiles, ray fan, GFPE range cut) and
    # wave-theoretic barrier insertion loss.
    generate_atmospheric_sound_speed_profiles,
    generate_atmospheric_ray_fan,
    generate_atmospheric_pe_range,
    generate_barrier_insertion_loss_methods,
    generate_barrier_geometry,
    generate_facade_elevation_geometry,
    generate_double_wall_geometry,
    generate_junction_plate_geometry,
    generate_insitu_setup_geometry,
    generate_dynamic_stiffness_rig_geometry,
    generate_diffusion_goniometer_geometry,
    generate_radiation_plate_geometry,
    generate_pp_probe_geometry,
    # Emission & electroacoustics: modulation sidebands, piston radiation
    # impedance, ISO 9614-1 field indicators, side-branch silencers, HVAC end
    # reflection, machine enclosures, phase decomposition and R 128 loudness.
    generate_modulation_distortion,
    generate_piston_radiation_impedance,
    generate_piston_baffle_geometry,
    generate_microphone_positions_hemisphere,
    generate_aperture_slit_geometry,
    generate_fdtd_domain_geometry,
    generate_field_indicators,
    # The emission practice figures: the two ISO 3744 corrections and their
    # caps, the ISO 3745 arrays, the per-segment partial powers of an
    # ISO 9614-2 scan, the p-p spacer trade, the radiation factor behind
    # ISO/TS 7849, the accuracy grades against an ISO 4871 declaration, and
    # the two BS.1770 quantities that are geometry rather than algebra.
    generate_k1_k2_corrections,
    generate_precision_positions_arrays,
    generate_partial_power_map,
    generate_reverberation_correction_terms,
    generate_spacer_bandwidth,
    generate_radiation_efficiency,
    generate_sound_power_grades_declaration,
    generate_true_peak_intersample,
    generate_channel_weight_map,
    generate_intensity_class,
    generate_silencer_side_branch,
    generate_silencer_insertion_loss,
    generate_silencer_selection,
    generate_silencer_extended_tube,
    generate_extended_tube_geometry,
    generate_silencer_chain_geometry,
    generate_helmholtz_branch_geometry,
    generate_quarter_wave_geometry,
    generate_plenum_geometry,
    generate_hvac_end_reflection,
    generate_hvac_elbow_flow_noise,
    generate_fan_sound_power,
    generate_duct_attenuation_elements,
    generate_duct_regenerated_noise,
    generate_duct_sheet_verification,
    generate_duct_path_cascade,
    generate_room_to_room_chain,
    generate_room_to_room_partitions,
    generate_duct_mode_cut_on,
    generate_enclosure_insertion_loss,
    generate_enclosure_required_tl,
    generate_phase_decomposition,
    generate_loudness_gating,
    generate_loudness_range,
    # Core-metrology figures (resampling, cepstral analysis, correlation,
    # cross-spectra, Golay recovery, synchronous averaging, runs test).
    generate_resampling_antialias,
    generate_cepstrum_variants,
    generate_lifter_split,
    generate_correlation_normalizations,
    generate_ir_alignment,
    generate_hilbert_envelope,
    generate_cross_spectral_density_delay,
    generate_coherent_output_snr,
    generate_golay_ir,
    generate_tsa_noise_reduction,
    generate_runs_test,
)


def generate_all(img_dir: str) -> None:
    """Generate every documentation figure for the currently active theme."""
    for func in _FIGURE_FUNCS:
        func(img_dir)


def generate_posters(output_dir: str) -> None:
    """Re-extract every animation poster from the already-rendered WebMs.

    Used by ``--posters`` (`make posters`) to refresh the stills without the
    slow re-encode of the clips themselves.
    """
    # `_extract_poster` trims the `.webm` off the text it is handed and returns
    # the poster path the same way, so the clips are named as text from here on.
    webms = sorted(str(webm) for webm in Path(output_dir).glob("anim_*.webm"))
    if not webms:
        msg = f"no anim_*.webm files found in {output_dir}; run `make animations` first"
        raise RuntimeError(msg)
    for webm in webms:
        poster = _extract_poster(webm, _poster_ss_for(webm))
        print(f"  {Path(webm).name} -> {Path(poster).name}")


_ANIMATIONS: dict[str, Callable[[str], None]] = {
    "anim_time_weighting": animate_time_weighting_ballistics,
    "anim_onset_detection": animate_onset_detection,
    "anim_instantaneous_intensity": animate_instantaneous_intensity,
    "anim_schroeder": animate_schroeder,
    "anim_fdtd_room_modes": animate_fdtd_room_modes,
    "anim_fdtd_barrier": animate_fdtd_barrier,
    "anim_fdtd_critical_angle": animate_fdtd_critical_angle,
    "anim_fdtd_ground_effect": animate_fdtd_ground_effect,
    "anim_fdtd_ducting": animate_fdtd_ducting,
    "anim_fdtd_diffusion": animate_fdtd_diffusion,
    "anim_fdtd_dispersion": animate_fdtd_dispersion,
    "anim_fdtd_duct_cut_on": animate_fdtd_duct_cut_on,
    "anim_fdtd_metadiffuser": animate_fdtd_metadiffuser,
    "anim_fdtd_pillar_hall": animate_fdtd_pillar_hall,
    "anim_fdtd_impedance_tube": animate_fdtd_impedance_tube,
    "anim_fdtd_transmission_tube": animate_fdtd_transmission_tube,
    "anim_standing_wave_tube": animate_standing_wave_tube,
    "anim_flanking_paths": animate_flanking_paths,
    "anim_intensity_scan_power": animate_intensity_scan_power,
    "anim_sweep_deconvolution": animate_sweep_deconvolution,
    "anim_specific_loudness": animate_specific_loudness,
    "anim_power_two_rooms": animate_power_two_rooms,
    "anim_comb_filtering": animate_comb_filtering,
    "anim_dynamic_stiffness_sweep": animate_dynamic_stiffness_sweep,
    "anim_fdtd_slit_absorber": animate_fdtd_slit_absorber,
    "anim_fdtd_expansion_chamber": animate_fdtd_expansion_chamber,
    "anim_fdtd_side_branch": animate_fdtd_side_branch,
    "anim_fdtd_absorption_placement": animate_fdtd_absorption_placement,
    "anim_fdtd_aperture_slit": animate_fdtd_aperture_slit,
    "anim_fdtd_refraction": animate_fdtd_refraction,
    "anim_elastic_plate_junction": animate_elastic_plate_junction,
    "anim_elastic_coincidence": animate_elastic_coincidence,
    "anim_elastic_halfspace_waves": animate_elastic_halfspace_waves,
    "anim_elastic_mode_conversion": animate_elastic_mode_conversion,
    "anim_elastic_radiation_efficiency": animate_elastic_radiation_efficiency,
    "anim_modulation_transfer": animate_modulation_transfer,
    "anim_loudness_gating": animate_loudness_gating,
    "anim_epnl_flyover": animate_epnl_flyover,
    "anim_image_source_buildup": animate_image_source_buildup,
    "anim_iso717_shift": animate_iso717_shift,
    "anim_block_vs_exponential": animate_block_vs_exponential,
    "anim_feedback_howl": animate_feedback_howl,
}


#: Cached field builders of the clips whose simulation dominates their cost,
#: keyed by clip name. Calling one fills its ``lru_cache`` in the current
#: process, which is what lets :func:`_render_anim_variants` fork the four
#: language/theme variants off a single field computation: the frame stacks
#: are never written after they are built, so the children share them
#: copy-on-write instead of paying for the simulation (or the memory) again.
_ANIM_FIELDS: dict[str, Callable[[], Any]] = {
    "anim_fdtd_room_modes": _room_mode_fields,
    "anim_fdtd_barrier": _barrier_fields,
    "anim_fdtd_critical_angle": _seabed_fields,
    "anim_fdtd_ground_effect": _ground_effect_fields,
    "anim_fdtd_ducting": _ducting_fields,
    "anim_fdtd_diffusion": _diffusion_fields,
    "anim_fdtd_dispersion": _dispersion_fields,
    "anim_fdtd_duct_cut_on": _duct_cut_on_fields,
    "anim_fdtd_metadiffuser": _metadiffuser_fields,
    "anim_fdtd_impedance_tube": _impedance_tube_fields,
    "anim_fdtd_transmission_tube": _transmission_tube_fields,
    "anim_fdtd_slit_absorber": _slit_absorber_fields,
    "anim_fdtd_expansion_chamber": _expansion_chamber_fields,
    "anim_fdtd_side_branch": _side_branch_fields,
    "anim_fdtd_absorption_placement": _absorption_placement_fields,
    "anim_fdtd_aperture_slit": _aperture_fields,
    "anim_fdtd_refraction": _refraction_fields,
    "anim_elastic_plate_junction": _plate_junction_fields,
    "anim_elastic_coincidence": _coincidence_fields,
    "anim_elastic_halfspace_waves": _halfspace_fields,
    "anim_elastic_mode_conversion": _mode_conversion_fields,
    "anim_elastic_radiation_efficiency": _radiation_efficiency_fields,
}

# A clip rename must not silently drop its field builder; fail fast.
if not _ANIM_FIELDS.keys() <= _ANIMATIONS.keys():
    _unknown_field = sorted(_ANIM_FIELDS.keys() - _ANIMATIONS.keys())
    msg = f"animation names not in _ANIMATIONS: {_unknown_field}"
    raise RuntimeError(msg)


def _render_anim_variant(clip: str, output_dir: str, lang: str, dark: bool) -> None:
    """Render one language/theme variant of *clip* (fork child entry)."""
    set_lang(lang)
    set_theme(dark)
    try:
        _ANIMATIONS[clip](output_dir)
    finally:
        plt.close("all")


def _render_anim_variants(clip: str, output_dir: str) -> None:
    """Render all four language/theme variants of one clip.

    Where the clip has a registered field builder and the platform can
    fork, the field is computed once here and the four variants then render
    concurrently in forked children that share it: the encoding of a
    600-frame field clip is the long pole, and four of them cost about as
    much wall time as one. Everything else (no builder, no fork) falls back
    to rendering the variants one after another in this process, which is
    what the pool workers of a full run do anyway.
    """
    import multiprocessing as mp

    builder = _ANIM_FIELDS.get(clip)
    if builder is None or "fork" not in mp.get_all_start_methods():
        for lang, dark in _VARIANTS:
            _render_anim_variant(clip, output_dir, lang, dark)
        return
    builder()
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_render_anim_variant, args=(clip, output_dir, lang, dark))
        for lang, dark in _VARIANTS
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    failed = [p.exitcode for p in procs if p.exitcode]
    if failed:
        msg = f"{clip}: {len(failed)} variant(s) failed (exit codes {failed})"
        raise RuntimeError(msg)


def _stamp_clips(clips: list[str], output_dir: str) -> None:
    """Record the fingerprint of every clip whose four variants were written.

    Both render paths end here -- the sequential one and the process pool of
    a full ``make animations`` -- because a stamp that only the one-clip path
    writes would leave the whole batch unstamped and the freshness check
    complaining about clips that were just re-rendered.

    Only a run that wrote into the committed image directory stamps: the
    renderer takes any output directory, and a render into a scratch one is
    for looking at, not a statement about what is committed.
    """
    import pathlib

    import animation_fingerprint

    committed = pathlib.Path(__file__).resolve().parents[2] / ".github" / "images"
    if not clips or pathlib.Path(output_dir).resolve() != committed:
        return
    animation_fingerprint.stamp(clips)


def generate_animations(
    output_dir: str, names: list[str] | None = None, *, variants: bool = False
) -> None:
    """Render the Tier-1 animations, by default in the active language/theme.

    ``names`` (clip stems, e.g. ``anim_schroeder``) restricts the run to a
    subset, used by ``--anim`` to re-render a single clip after review
    fixes without paying for the whole batch. With ``variants`` each clip is
    rendered in all four language x theme variants off one field
    computation (:func:`_render_anim_variants`) instead of once in whatever
    language and theme the caller has set.

    A four-variant render also stamps the clip's fingerprint into
    ``scripts/animation_fingerprints.txt``, which is how
    ``scripts/check_animation_freshness.py`` can later tell that the code
    drawing a committed clip has moved on without it. The stamp is written
    here, with the files it describes, so a re-render cannot forget it; a
    single-variant render does not stamp, because it leaves the other three
    variants of that clip as they were.

    The stamping is in a ``finally`` because a batch does not fail as a
    whole: a clip that raises halfway through leaves the ones before it
    complete on disk, and dropping their stamps with the error would have
    the freshness check call finished clips stale until somebody re-rendered
    them for nothing. The pool path already stamps what succeeded before it
    reports the failures (:func:`_generate_animations_parallel`); this is the
    same bargain for the sequential one, and the render error still leaves.
    """
    import shutil

    if shutil.which("ffmpeg") is None:
        msg = (
            "ffmpeg was not found on PATH; it is required to encode the "
            "animation WebM/GIF outputs. Install ffmpeg and retry."
        )
        raise RuntimeError(msg)
    if names:
        unknown = sorted(set(names) - _ANIMATIONS.keys())
        if unknown:
            available = ", ".join(sorted(_ANIMATIONS))
            msg = f"unknown animation(s) {unknown}; available: {available}"
            raise SystemExit(msg)
        clips = list(names)
    else:
        clips = list(_ANIMATIONS)
    stamped: list[str] = []
    try:
        for clip in clips:
            if variants:
                print(f"--- Generating {clip} (4 variants) ---")
                _render_anim_variants(clip, output_dir)
                stamped.append(clip)
            else:
                _ANIMATIONS[clip](output_dir)
    finally:
        _stamp_clips(stamped, output_dir)


# ====================================================================# Command line / parallel figure generation
# ---------------------------------------------------------------------------
# Every asset is produced four times: light/dark theme x English/Spanish
# ("_dark" / "_es" / "_es_dark" suffixes) so both site languages follow the
# user's mode. The figures are independent (each renders from its own seeded
# data into its own per-variant files), so the (figure, language, theme)
# tasks are distributed over a process pool. Language and theme are applied
# inside the worker before each figure runs, exactly as the sequential loop
# did, so the output bytes are identical either way.
# ====================================================================
_VARIANTS: tuple[tuple[str, bool], ...] = (
    ("en", False),
    ("en", True),
    ("es", False),
    ("es", True),
)

# Figures whose expensive signal analysis is memoised with ``lru_cache``
# across the four language/theme variants (the ECMA-418-2 / ISO 532
# psychoacoustic computations). The caches live per process, so these render
# all four variants as ONE task in a single worker -- four separate tasks
# would recompute the same analysis in four workers (~150 s of duplicated
# CPU). Every other figure is cheap to recompute and parallelises per
# variant.
_GROUPED_FIGURES = frozenset(
    {
        "loudness_models_comparison",
        "sottek_specific_loudness",
        "tonality_roughness_demo",
        "moore_glasberg_time_loudness",
        "fluctuation_strength",
        "metadiffuser_ntff_polar",
    }
)

# Approximate task cost in seconds (measured sequentially on a 12-core dev
# box; for the grouped figures the value is the whole four-variant task).
# Used only to submit the heaviest tasks to the pool first so none of them
# lands at the tail and stretches the run; unlisted figures are treated as
# fast and staleness is harmless.
_FIGURE_WEIGHTS: dict[str, float] = {
    "metadiffuser_ntff_polar": 260.0,
    "loudness_models_comparison": 25.5,
    "tonality_roughness_demo": 19.7,
    "sottek_specific_loudness": 4.1,
    "filter_responses": 2.8,
    "moore_glasberg_time_loudness": 2.6,
    "numerical_propagation": 2.5,
    "gaussian_beam_caustic": 8.0,
    "fluctuation_strength": 2.2,
    "weighting_responses": 1.1,
    "special_weighting_responses": 2.6,
    "sti_curve": 1.7,
    "schroeder_decay": 1.3,
    "source_distance_bias": 9.0,
    "excitation_signals": 1.2,
    "crossover_plot": 1.1,
    "fdtd_room_modes": 2.0,
}

# A registry rename must not silently degroup a cached figure (a 4x
# recompute) or drop its scheduling weight; fail fast on import instead.
_REGISTRY_NAMES = frozenset(f.__name__.removeprefix("generate_") for f in _FIGURE_FUNCS)
if not (_GROUPED_FIGURES | _FIGURE_WEIGHTS.keys()) <= _REGISTRY_NAMES:
    _unknown = sorted((_GROUPED_FIGURES | _FIGURE_WEIGHTS.keys()) - _REGISTRY_NAMES)
    msg = f"figure names not in _FIGURE_FUNCS: {_unknown}"
    raise RuntimeError(msg)


def _run_figure_task(
    func_name: str, variants: tuple[tuple[str, bool], ...], img_dir: str
) -> str:
    """Render one figure in the given language/theme variants (worker entry).

    The prologue applies each variant's language and theme to this worker's
    module globals, exactly like the sequential loop does before calling the
    generator; the epilogue closes every figure so a task cannot leak pyplot
    state (or figure memory) into the next task that reuses the process.
    """
    func: Callable[[str], None] = getattr(sys.modules[__name__], func_name)
    try:
        for lang, dark in variants:
            set_lang(lang)
            set_theme(dark)
            func(img_dir)
    finally:
        plt.close("all")
    return func_name


def _default_jobs() -> int:
    """Default worker count: the available cores minus two, capped at 8.

    Two cores are left free so the machine (or the CI runner's other duties)
    stays responsive, and the cap bounds peak memory: every worker holds one
    figure's data arrays plus a matplotlib canvas (~a few hundred MB for the
    heaviest compute figures).
    """
    cpus = os.process_cpu_count() or 2
    return max(1, min(cpus - 2, 8))


def _generate_figures_parallel(
    img_dir: str, funcs: list[Callable[[str], None]], jobs: int
) -> None:
    """Render ``funcs`` in all four variants on a ``jobs``-wide process pool.

    Workers are spawned (not forked) so each starts from a pristine
    interpreter: a fresh import of this module pins the numerical thread
    pools and leaves no inherited pyplot/rcParams state, keeping every task
    bit-reproducible regardless of scheduling order.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

    tasks: list[tuple[str, tuple[tuple[str, bool], ...]]] = []
    for func in funcs:
        if func.__name__.removeprefix("generate_") in _GROUPED_FIGURES:
            tasks.append((func.__name__, _VARIANTS))
        else:
            tasks.extend((func.__name__, (variant,)) for variant in _VARIANTS)
    # Heaviest tasks first: they hit idle workers immediately instead of
    # serialising the tail of the run.
    tasks.sort(key=lambda t: -_FIGURE_WEIGHTS.get(t[0].removeprefix("generate_"), 0.0))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {
            pool.submit(_run_figure_task, name, variants, img_dir): (name, variants)
            for name, variants in tasks
        }
        _done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for future in not_done:
            future.cancel()
    # The pool has drained here (the `with` block shut it down), so every
    # non-cancelled future is settled; scanning them all also aggregates
    # failures from tasks that were still running when wait() returned on
    # the first exception.
    failures: list[str] = []
    cancelled = 0
    for future, (name, variants) in futures.items():
        if future.cancelled():
            cancelled += 1
            continue
        if (exc := future.exception()) is not None:
            labels = ", ".join(
                f"{lang} {'dark' if dark else 'light'}" for lang, dark in variants
            )
            # The executor chains the worker's formatted traceback as
            # __cause__; keep it, or a failure only says what raised, not
            # where in the generators.
            detail = f"{exc!r}"
            if exc.__cause__ is not None:
                detail += f"\n{exc.__cause__}"
            failures.append(f"{name} [{labels}]: {detail}")
    if failures:
        skipped = (
            f"\n  ({cancelled} queued tasks cancelled, not attempted)"
            if cancelled
            else ""
        )
        raise RuntimeError(
            "figure generation failed:\n  " + "\n  ".join(sorted(failures)) + skipped
        )


# Approximate per-clip render cost (all four variants, the FDTD simulation
# amortised once per clip through the ``lru_cache``). Used only to submit the
# heaviest clips to the pool first so a long-pole FDTD clip does not land at
# the tail; the exact values are irrelevant to correctness. The wave-field
# clips dominate (a full 2D simulation plus four WebM encodes each) and the
# weights track their captured frame counts and mesh sizes -- the SOFAR duct
# at 3 793 frames and the sub-millimetre slit absorber, whose simulation
# alone runs three quarters of an hour, are the long poles. The schematics
# are cheap and unlisted clips are treated as light.
_ANIM_WEIGHTS: dict[str, float] = {
    "anim_fdtd_room_modes": 1500.0,
    "anim_fdtd_barrier": 1000.0,
    "anim_fdtd_critical_angle": 1200.0,
    "anim_fdtd_ground_effect": 700.0,
    "anim_fdtd_ducting": 1600.0,
    "anim_fdtd_diffusion": 750.0,
    "anim_fdtd_dispersion": 480.0,
    "anim_fdtd_metadiffuser": 540.0,
    "anim_fdtd_pillar_hall": 200.0,
    "anim_standing_wave_tube": 130.0,
    "anim_sweep_deconvolution": 90.0,
    "anim_power_two_rooms": 80.0,
    "anim_specific_loudness": 80.0,
    "anim_comb_filtering": 70.0,
    "anim_dynamic_stiffness_sweep": 70.0,
    "anim_flanking_paths": 70.0,
    "anim_instantaneous_intensity": 65.0,
    "anim_intensity_scan_power": 60.0,
    "anim_onset_detection": 55.0,
    "anim_time_weighting": 55.0,
    "anim_schroeder": 55.0,
    "anim_fdtd_impedance_tube": 300.0,
    "anim_fdtd_transmission_tube": 200.0,
    "anim_fdtd_slit_absorber": 1500.0,
    "anim_fdtd_aperture_slit": 800.0,
    "anim_fdtd_refraction": 900.0,
    "anim_fdtd_expansion_chamber": 250.0,
    "anim_fdtd_side_branch": 780.0,
    "anim_fdtd_absorption_placement": 2100.0,
    "anim_fdtd_duct_cut_on": 260.0,
    "anim_elastic_plate_junction": 420.0,
    "anim_elastic_coincidence": 640.0,
    "anim_elastic_halfspace_waves": 320.0,
    "anim_elastic_mode_conversion": 820.0,
    "anim_elastic_radiation_efficiency": 700.0,
    "anim_modulation_transfer": 95.0,
    "anim_loudness_gating": 90.0,
    "anim_epnl_flyover": 85.0,
    "anim_image_source_buildup": 110.0,
    "anim_iso717_shift": 60.0,
    "anim_block_vs_exponential": 75.0,
    "anim_feedback_howl": 65.0,
}

# A clip rename must not silently drop its scheduling weight; fail fast.
if not _ANIM_WEIGHTS.keys() <= _ANIMATIONS.keys():
    _unknown_anim = sorted(_ANIM_WEIGHTS.keys() - _ANIMATIONS.keys())
    msg = f"animation names not in _ANIMATIONS: {_unknown_anim}"
    raise RuntimeError(msg)


def _run_anim_task(clip: str, img_dir: str) -> str:
    """Render one clip in all four language/theme variants (worker entry).

    All four variants render in the SAME worker so a clip's expensive FDTD
    simulation -- memoised with ``lru_cache`` -- is computed once and reused
    across the variants, exactly as the sequential loop's per-process cache
    did. Splitting the variants across workers would recompute the field
    four times. Each variant applies its language/theme to this worker's
    module globals before rendering, and the epilogue closes every figure so
    pyplot state cannot leak into the next clip that reuses the process.
    """
    func = _ANIMATIONS[clip]
    try:
        for lang, dark in _VARIANTS:
            set_lang(lang)
            set_theme(dark)
            func(img_dir)
    finally:
        plt.close("all")
        # The banner's 2.5 mm frame stack is ~2.5 GB; do not let it stay
        # pinned in this worker while it renders the next clip.
        _pillar_fields.cache_clear()
    return clip


def _generate_animations_parallel(img_dir: str, clips: list[str], jobs: int) -> None:
    """Render ``clips`` (all four variants each) on a ``jobs``-wide pool.

    Mirrors :func:`_generate_figures_parallel`: spawned workers each import a
    pristine module (numerical thread pools pinned to one thread), one grouped
    task per clip, heaviest clips submitted first so the long-pole FDTD field
    clips are not left to serialise at the tail of the run.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

    tasks = sorted(clips, key=lambda c: -_ANIM_WEIGHTS.get(c, 0.0))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {pool.submit(_run_anim_task, clip, img_dir): clip for clip in tasks}
        _done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for future in not_done:
            future.cancel()
    failures: list[str] = []
    cancelled = 0
    for future, clip in futures.items():
        if future.cancelled():
            cancelled += 1
            continue
        if (exc := future.exception()) is not None:
            detail = f"{exc!r}"
            if exc.__cause__ is not None:
                detail += f"\n{exc.__cause__}"
            failures.append(f"{clip}: {detail}")
    _stamp_clips(
        [
            clip
            for future, clip in futures.items()
            if not future.cancelled() and future.exception() is None
        ],
        img_dir,
    )
    if failures:
        skipped = (
            f"\n  ({cancelled} queued clips cancelled, not attempted)"
            if cancelled
            else ""
        )
        raise RuntimeError(
            "animation generation failed:\n  " + "\n  ".join(sorted(failures)) + skipped
        )


def _select_figures(names: list[str] | None) -> list[Callable[[str], None]]:
    """Resolve ``--figure`` names (with or without the ``generate_`` prefix)."""
    if not names:
        return list(_FIGURE_FUNCS)
    by_name = {f.__name__.removeprefix("generate_"): f for f in _FIGURE_FUNCS}
    selected = []
    for raw in names:
        name = raw.removeprefix("generate_")
        if name not in by_name:
            available = ", ".join(sorted(by_name))
            msg = f"unknown figure {raw!r}; available: {available}"
            raise SystemExit(msg)
        selected.append(by_name[name])
    return selected


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: figures by default, clips behind ``--animations``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate the documentation figures (and animations) "
        "in all four language x theme variants."
    )
    parser.add_argument(
        "--animations",
        action="store_true",
        help="render only the Tier-1 animation clips (slow ffmpeg encoding, "
        "kept out of the default figure run)",
    )
    parser.add_argument(
        "--posters",
        action="store_true",
        help="re-extract only the animation poster stills from the "
        "already-rendered WebM files (no clip re-encoding)",
    )
    parser.add_argument(
        "--anim",
        action="append",
        default=None,
        metavar="NAME",
        help="with --animations, render only this clip (repeatable; use "
        "the output stem, e.g. --anim anim_schroeder)",
    )
    parser.add_argument(
        "--all",
        dest="do_all",
        action="store_true",
        help="render both the figures and the animations",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="worker processes for the figures (default: cores minus two, "
        "capped at 8; 1 renders sequentially in-process)",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=None,
        metavar="NAME",
        help="render only this figure (repeatable; the generate_ prefix is "
        "optional, e.g. --figure loudness_models_comparison)",
    )
    args = parser.parse_args(argv)

    img_dir = ".github/images"
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    do_figs = not (args.animations or args.posters) or args.do_all
    do_anim = args.animations or args.do_all

    if args.anim and not do_anim:
        parser.error("--anim requires --animations (or --all)")
    if args.posters and not do_anim:
        print("--- Re-extracting animation posters ---")
        generate_posters(img_dir)
    jobs = args.jobs if args.jobs is not None else _default_jobs()
    if jobs < 1:
        parser.error("--jobs must be >= 1")

    if do_figs:
        funcs = _select_figures(args.figure)
        if jobs == 1:
            for lang, dark in _VARIANTS:
                set_lang(lang)
                set_theme(dark)
                print(
                    f"--- Generating {lang} {'dark' if dark else 'light'} theme figures ---"
                )
                for func in funcs:
                    func(img_dir)
        else:
            print(
                f"--- Generating figures ({len(funcs)} x 4 variants, {jobs} jobs) ---"
            )
            _generate_figures_parallel(img_dir, funcs, jobs)

    if do_anim:
        if args.anim or jobs == 1:
            # A targeted re-render (``--anim``) and an explicit ``--jobs 1``
            # walk the clips one at a time: each clip's field is simulated
            # once and its four language/theme variants then render off that
            # single computation (see _render_anim_variants).
            generate_animations(img_dir, args.anim, variants=True)
        else:
            import shutil

            if shutil.which("ffmpeg") is None:
                msg = (
                    "ffmpeg was not found on PATH; it is required to encode "
                    "the animation WebM/GIF outputs. Install ffmpeg and retry."
                )
                raise RuntimeError(msg)
            clips = list(_ANIMATIONS)
            print(
                f"--- Generating animations ({len(clips)} clips "
                f"x 4 variants, {jobs} jobs) ---"
            )
            _generate_animations_parallel(img_dir, clips, jobs)

    print("Graphics generated successfully.")


if __name__ == "__main__":
    main()
