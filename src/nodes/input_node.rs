use super::{
    output::Out,
    input::In,
};
use crate::conn::connection::Connection;

pub struct InputNode<N: In> {
    val: f32,
    next_nodes: Vec<Connection<Self,N>>,
}

impl <N: In>InputNode<N> {
    fn new() -> Self {
        InputNode{
            val: 0.0,
            next_nodes: vec![],
        }
    }
}

impl <N: In> In for InputNode<N> {
    fn recieve(&mut self, x: f32) {
        self.val = x;
    }
}

impl <N: In> Out for InputNode<N> {
    #[inline]
    fn activation(&self, x: f32) -> f32 {
        x
    }
    #[inline]
    fn local_gradient(&self, err: f32) -> f32 {
        err 
    }
    
    fn forward_prop(&mut self) {
        for n in &self.next_nodes {
            n.to.borrow_mut().recieve(self.activation(self.val) * n.weight);
        }
    }
}