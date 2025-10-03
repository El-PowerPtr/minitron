pub trait Connect {
    fn forward_prop(&mut self, val: f32);
}