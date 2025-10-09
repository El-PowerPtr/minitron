pub trait Out{
    fn activation(&self, x: f32) -> f32;
    fn local_gradient(&self, err: f32) ->f32;
    fn forward_prop(&mut self, val: f32);
}

    

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid_derivative(x: f32) -> f32 {
    let neg_exp = (-x).exp();
    let base = 1.0 + neg_exp;
    - neg_exp / (base * base)
}

#[inline]
pub fn error(actual: f32, expected: f32) -> f32 {
    (expected - actual) * (expected - actual)
}
    
#[inline]
pub fn error_der(actual: f32, expected:f32) -> f32 {
    -2.0 * (expected - actual)
}