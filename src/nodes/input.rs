use super::output::*;
use crate::conn::connect::Connect;
use crate::conn::connection::Connection;

pub trait In {
    fn recieve(&mut self, x: f32);
}

pub struct InputNode<N: In> {
    next_nodes: Vec<Connection<Self,N>>,
}

impl <N: In> In for InputNode<N> {
    fn recieve(&mut self, x: f32) {
        self.forward_prop(x)
    }
}

impl <N: In> Out for InputNode<N> {
    fn activation(&self, x: f32) -> f32 {
        x
    }
}

impl <N: In> Connect for InputNode<N> {
    fn forward_prop(&mut self, x: f32) {
        for n in &self.next_nodes {
            n.to.borrow_mut().recieve(self.activation(x) * n.weight);
        }
    }
}