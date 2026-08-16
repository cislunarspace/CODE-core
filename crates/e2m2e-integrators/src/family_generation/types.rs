//! 统一轨道族生成的内部类型。

#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    pub(crate) mu: f64,
    pub(crate) characteristic_length_km: f64,
    pub(crate) secondary_radius_km: f64,
    pub(crate) rtol: f64,
    pub(crate) atol: f64,
    pub(crate) max_step: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) enum Spec {
    Halo {
        point: u8,
        max_amplitude_km: f64,
        member_limit: usize,
    },
    Nrho {
        point: u8,
        north_south: u8,
        perilune_height_max_km: f64,
        member_limit: usize,
    },
    Axial {
        point: u8,
        max_amplitude_km: f64,
        member_limit: usize,
    },
    Lissajous {
        point: u8,
        amplitude_in_km: f64,
        amplitude_out_km: f64,
        phase_in: f64,
        phase_out: f64,
        member_limit: usize,
        n_periods: usize,
    },
    Triangular {
        family_type: &'static str,
        point: u8,
        min_amplitude_km: f64,
        max_amplitude_km: f64,
        member_limit: usize,
        direction: &'static str,
        match_tolerance_km: f64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Member {
    pub(crate) states: Vec<[f64; 6]>,
    pub(crate) times: Vec<f64>,
    pub(crate) period: Option<f64>,
    pub(crate) closure_error: Option<f64>,
    pub(crate) amplitude_km: Option<f64>,
    pub(crate) perilune_height_km: Option<f64>,
    pub(crate) sampling_fraction: Option<f64>,
    pub(crate) jacobi_drift: Option<f64>,
    pub(crate) newton_iterations: Option<usize>,
    pub(crate) tangent_system_rank: Option<usize>,
    pub(crate) tangent_system_condition: Option<f64>,
    pub(crate) augmented_system_rank: Option<usize>,
    pub(crate) augmented_system_condition: Option<f64>,
    pub(crate) step_size: Option<f64>,
}

impl Member {
    pub(crate) fn periodic(
        state: [f64; 6],
        period: f64,
        closure_error: f64,
        jacobi_drift: Option<f64>,
    ) -> Self {
        Self {
            states: vec![state],
            times: vec![0.0],
            period: Some(period),
            closure_error: Some(closure_error),
            amplitude_km: None,
            perilune_height_km: None,
            sampling_fraction: None,
            jacobi_drift,
            newton_iterations: None,
            tangent_system_rank: None,
            tangent_system_condition: None,
            augmented_system_rank: None,
            augmented_system_condition: None,
            step_size: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Outcome {
    pub(crate) family_type: &'static str,
    pub(crate) periodicity: &'static str,
    pub(crate) status: &'static str,
    pub(crate) cause: &'static str,
    pub(crate) message: String,
    pub(crate) requested_members: usize,
    pub(crate) members: Vec<Member>,
}

impl Outcome {
    pub(crate) fn converged(
        family_type: &'static str,
        periodicity: &'static str,
        requested_members: usize,
        members: Vec<Member>,
    ) -> Self {
        Self {
            family_type,
            periodicity,
            status: "converged",
            cause: "none",
            message: "轨道族生成完成".to_string(),
            requested_members,
            members,
        }
    }

    pub(crate) fn soft_failure(
        family_type: &'static str,
        periodicity: &'static str,
        requested_members: usize,
        members: Vec<Member>,
        status: &'static str,
        cause: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            family_type,
            periodicity,
            status,
            cause,
            message: message.into(),
            requested_members,
            members,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PeriodicOrbit {
    pub(crate) state: [f64; 6],
    pub(crate) period: f64,
    pub(crate) closure_error: f64,
}
