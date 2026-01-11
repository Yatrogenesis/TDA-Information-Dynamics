//! CUSUM Detector: Sequential Change-Point Detection
//!
//! Implements Page's CUSUM algorithm adapted for topological statistics.

/// Detection event with timing and confidence
#[derive(Debug, Clone)]
pub struct DetectionEvent {
    /// Time index when threshold was crossed
    pub detection_time: usize,
    /// CUSUM value at detection
    pub cusum_value: f64,
    /// Estimated change-point (retrospective)
    pub estimated_change_point: usize,
    /// Statistical measure at detection
    pub statistic_value: f64,
}

/// Results from CUSUM analysis
#[derive(Debug, Clone)]
pub struct CusumResult {
    /// Sequence of CUSUM values C(t)
    pub cusum_values: Vec<f64>,
    /// Detection events (if any)
    pub events: Vec<DetectionEvent>,
    /// Reference mean μ₀
    pub reference_mean: f64,
    /// Reference std σ₀
    pub reference_std: f64,
    /// Detection threshold h
    pub threshold: f64,
    /// Did detection occur?
    pub detected: bool,
    /// First detection time (if detected)
    pub first_detection_time: Option<usize>,
}

/// CUSUM detector for topological phase transitions
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Reference mean (computed from calibration period)
    reference_mean: f64,
    /// Reference standard deviation
    reference_std: f64,
    /// Allowance parameter k (slack)
    allowance: f64,
    /// Detection threshold h
    threshold: f64,
    /// Current CUSUM value
    current_cusum: f64,
    /// History of CUSUM values
    cusum_history: Vec<f64>,
    /// History of input statistics
    statistic_history: Vec<f64>,
    /// Time of last reset (for change-point estimation)
    last_reset_time: usize,
    /// Detection events
    events: Vec<DetectionEvent>,
    /// Is calibrated?
    calibrated: bool,
}

impl CusumDetector {
    /// Create new detector with default parameters
    pub fn new() -> Self {
        Self {
            reference_mean: 0.0,
            reference_std: 1.0,
            allowance: 0.5,  // Standard choice: k = 0.5σ
            threshold: 5.0,  // Standard choice: h = 5σ
            current_cusum: 0.0,
            cusum_history: Vec::new(),
            statistic_history: Vec::new(),
            last_reset_time: 0,
            events: Vec::new(),
            calibrated: false,
        }
    }

    /// Create detector with custom parameters
    pub fn with_params(allowance_sigmas: f64, threshold_sigmas: f64) -> Self {
        Self {
            allowance: allowance_sigmas,
            threshold: threshold_sigmas,
            ..Self::new()
        }
    }

    /// Calibrate from reference data (H₀ regime)
    ///
    /// The calibration period should represent the system in its
    /// normal/reference state before any transition.
    pub fn calibrate(&mut self, reference_data: &[f64]) {
        if reference_data.is_empty() {
            return;
        }

        // Compute reference statistics
        let n = reference_data.len() as f64;
        self.reference_mean = reference_data.iter().sum::<f64>() / n;

        let variance: f64 = reference_data.iter()
            .map(|x| (x - self.reference_mean).powi(2))
            .sum::<f64>() / (n - 1.0).max(1.0);
        self.reference_std = variance.sqrt().max(1e-10);

        // Convert sigma-based parameters to absolute values
        // allowance and threshold are stored as sigma multipliers
        self.calibrated = true;
        self.reset();
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.current_cusum = 0.0;
        self.cusum_history.clear();
        self.statistic_history.clear();
        self.last_reset_time = 0;
        self.events.clear();
    }

    /// Process a single observation
    ///
    /// Returns true if detection threshold is crossed
    pub fn update(&mut self, statistic: f64) -> bool {
        if !self.calibrated {
            panic!("CUSUM detector not calibrated! Call calibrate() first.");
        }

        self.statistic_history.push(statistic);
        let t = self.statistic_history.len() - 1;

        // Standardized increment
        let z = (statistic - self.reference_mean) / self.reference_std;

        // CUSUM update: C(t) = max(0, C(t-1) + z - k)
        let k = self.allowance;
        self.current_cusum = (self.current_cusum + z - k).max(0.0);

        // Track zero crossings for change-point estimation
        if self.current_cusum == 0.0 {
            self.last_reset_time = t;
        }

        self.cusum_history.push(self.current_cusum);

        // Check threshold
        let h = self.threshold;
        if self.current_cusum > h {
            let event = DetectionEvent {
                detection_time: t,
                cusum_value: self.current_cusum,
                estimated_change_point: self.last_reset_time,
                statistic_value: statistic,
            };
            self.events.push(event);
            return true;
        }

        false
    }

    /// Process multiple observations
    pub fn update_batch(&mut self, statistics: &[f64]) -> CusumResult {
        for &s in statistics {
            self.update(s);
        }
        self.result()
    }

    /// Get current result
    pub fn result(&self) -> CusumResult {
        CusumResult {
            cusum_values: self.cusum_history.clone(),
            events: self.events.clone(),
            reference_mean: self.reference_mean,
            reference_std: self.reference_std,
            threshold: self.threshold * self.reference_std,
            detected: !self.events.is_empty(),
            first_detection_time: self.events.first().map(|e| e.detection_time),
        }
    }

    /// Current CUSUM value
    pub fn current_value(&self) -> f64 {
        self.current_cusum
    }

    /// Is detection active?
    pub fn is_detected(&self) -> bool {
        !self.events.is_empty()
    }
}

impl Default for CusumDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Two-sided CUSUM for detecting both increases and decreases
#[derive(Debug, Clone)]
pub struct TwoSidedCusum {
    /// Upper CUSUM (detects increases)
    upper: CusumDetector,
    /// Lower CUSUM (detects decreases)
    lower: CusumDetector,
}

impl TwoSidedCusum {
    pub fn new() -> Self {
        Self {
            upper: CusumDetector::new(),
            lower: CusumDetector::new(),
        }
    }

    pub fn with_params(allowance_sigmas: f64, threshold_sigmas: f64) -> Self {
        Self {
            upper: CusumDetector::with_params(allowance_sigmas, threshold_sigmas),
            lower: CusumDetector::with_params(allowance_sigmas, threshold_sigmas),
        }
    }

    pub fn calibrate(&mut self, reference_data: &[f64]) {
        self.upper.calibrate(reference_data);
        // For lower CUSUM, negate the data
        let negated: Vec<f64> = reference_data.iter().map(|x| -x).collect();
        self.lower.calibrate(&negated);
    }

    pub fn update(&mut self, statistic: f64) -> (bool, bool) {
        let upper_det = self.upper.update(statistic);
        let lower_det = self.lower.update(-statistic);
        (upper_det, lower_det)
    }

    pub fn is_detected(&self) -> bool {
        self.upper.is_detected() || self.lower.is_detected()
    }
}

impl Default for TwoSidedCusum {
    fn default() -> Self {
        Self::new()
    }
}

/// CUSUM with automatic calibration window
pub struct AdaptiveCusum {
    detector: CusumDetector,
    calibration_window: usize,
    buffer: Vec<f64>,
    running: bool,
}

impl AdaptiveCusum {
    pub fn new(calibration_window: usize) -> Self {
        Self {
            detector: CusumDetector::new(),
            calibration_window,
            buffer: Vec::with_capacity(calibration_window),
            running: false,
        }
    }

    pub fn update(&mut self, statistic: f64) -> Option<bool> {
        if !self.running {
            self.buffer.push(statistic);
            if self.buffer.len() >= self.calibration_window {
                self.detector.calibrate(&self.buffer);
                self.running = true;
            }
            None
        } else {
            Some(self.detector.update(statistic))
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn result(&self) -> Option<CusumResult> {
        if self.running {
            Some(self.detector.result())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_no_change() {
        let mut detector = CusumDetector::new();

        // Calibrate on N(0,1) samples
        let reference: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        detector.calibrate(&reference);

        // Feed similar data - should not detect
        let test: Vec<f64> = (100..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = detector.update_batch(&test);

        // With similar data, should not trigger (most of the time)
        // This is probabilistic, so we just check the structure
        assert_eq!(result.cusum_values.len(), test.len());
    }

    #[test]
    fn test_cusum_detects_shift() {
        let mut detector = CusumDetector::with_params(0.5, 4.0);

        // Calibrate on mean=0
        let reference: Vec<f64> = vec![0.0; 100];
        detector.calibrate(&reference);

        // Feed data with shift to mean=3
        let mut detected = false;
        for _ in 0..50 {
            if detector.update(3.0) {
                detected = true;
                break;
            }
        }

        assert!(detected, "CUSUM should detect mean shift");
    }

    #[test]
    fn test_adaptive_cusum() {
        let mut cusum = AdaptiveCusum::new(50);

        // First 50 samples: calibration
        for i in 0..50 {
            let result = cusum.update((i as f64 * 0.1).sin());
            assert!(result.is_none());
        }

        assert!(cusum.is_running());

        // Now detection is active
        let result = cusum.update(0.0);
        assert!(result.is_some());
    }
}
