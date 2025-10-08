pub trait Learn {
    fn learn(&mut self);
    fn get_feedback(&mut self, feedback: f32);
}

pub trait BackProp {
    fn send_feedback(&mut self);
}