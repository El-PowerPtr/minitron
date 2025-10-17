use super::output::Out;
use crate::MultiRef;

pub trait In {
    fn recieve(&mut self, x: f32);
}

pub trait Connected<I: Out>{
    fn get_connected(&mut self,node: MultiRef<I>);
}