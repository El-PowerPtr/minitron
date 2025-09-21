use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell
};

use super::output::*;

pub (super) trait In {
    fn activation(&self, x: f32) -> f32;
    fn recieve(&mut self, x: f32);
}

pub (super) type InRef<N> = Rc<RefCell<Box<N>>>;

pub (super) struct InputNode<N: In> {
    next_nodes: Vec<(f32, InRef<N>)>,
}

impl <N: In> In for InputNode<N> {
    fn activation(&self, x: f32) -> f32 {
        x
    }
    fn recieve(&mut self, x: f32) {
        self.forward_prop(x)
    }
}

impl <N: In> Connect<N> for InputNode<N> {
    fn connect(&mut self, weight: f32, node: InRef<N>) {
        self.next_nodes.push((weight, node.clone()))
    }
    
    fn forward_prop(&mut self, x: f32) {
        for n in &self.next_nodes {
            n.1.borrow_mut().recieve(self.activation(x) * n.0);
        }
    }
}