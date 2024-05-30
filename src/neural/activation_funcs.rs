use std::f32::consts::E;

pub type Func = fn(f64) -> f64;
pub type ActivationFunc = (Func, Func);

pub const IDENTITY: ActivationFunc = (|x: f64| x, |_: f64| 1.);
pub const RELU: ActivationFunc = (
    |x: f64| {
        return x.max(0.);
    },
    |x: f64| {
        if x > 0. {
            return 1.;
        }
        return 0.;
    },
);

const SIGMOID_GX: Func = |x| 1. / (1. + (E.powf(x as f32) as f64));
pub const SIGMOID: ActivationFunc = (SIGMOID_GX, |x: f64| SIGMOID_GX(x)*(1. -SIGMOID_GX(x)));

#[cfg(test)]
mod tests {
    use super::*;
    // Relu tests
    #[test]
    fn relu_gt_0() {
        assert_eq!(RELU.0(1.), 1.);
    }
    #[test]
    fn relu_eq_0() {
        assert_eq!(RELU.0(0.), 0.);
    }
    #[test]
    fn relu_lt_0() {
        assert_eq!(RELU.0(-1.), 0.);
    }

    // Identity tests
    #[test]
    fn identity_gt_0() {
        assert_eq!(IDENTITY.0(1.), 1.);
    }
    #[test]
    fn identity_eq_0() {
        assert_eq!(IDENTITY.0(0.), 0.);
    }
    #[test]
    fn identity_lt_0() {
        assert_eq!(IDENTITY.0(-1.), -1.);
    }
}
