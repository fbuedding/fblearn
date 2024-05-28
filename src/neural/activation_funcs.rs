pub trait ActivationF {
    fn f(&self, x: f64) -> f64;
    fn f_derivative(&self, x: f64) -> f64;
}

pub struct Relu;
impl ActivationF for Relu {
    fn f(&self, x: f64) -> f64 {
        return x.max(0.);
    }

    fn f_derivative(&self, x: f64) -> f64 {
        if x > 0. {
            return 1.;
        }
        return 0.;
    }
}
pub struct Identity;
impl ActivationF for Identity {
    fn f(&self, x: f64) -> f64 {
        return x;
    }

    fn f_derivative(&self, _: f64) -> f64 {
        return 1.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Relu tests
    #[test]
    fn relu_gt_0() {
        let relu = Relu;
        assert_eq!(relu.f(1.), 1.);
    }
    #[test]
    fn relu_eq_0() {
        let relu = Relu;
        assert_eq!(relu.f(0.), 0.);
    }
    #[test]
    fn relu_lt_0() {
        let relu = Relu;
        assert_eq!(relu.f(-1.), 0.);
    }

    // Identity tests
    #[test]
    fn identity_gt_0() {
        let identity = Identity;
        assert_eq!(identity.f(1.), 1.);
    }
    #[test]
    fn identity_eq_0() {
        let identity = Identity;
        assert_eq!(identity.f(0.), 0.);
    }
    #[test]
    fn identity_lt_0() {
        let identity = Identity;
        assert_eq!(identity.f(-1.), -1.);
    }
}
