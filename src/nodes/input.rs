use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell
};

pub (mod) trait In {
    fn comp_in(&self) -> f32;
    fn recieve(x: f32);
}

type InRef<N: In> = Rc<RefCell<Box<N>>>;

pub (mod) struct InputNode<N: In>: In + Out + Connect {
    next_nodes: Vec<(f32, InRef<N>)>,
}

impl <N: In> In for InputNode<N> {
    fn activation(x: f32) -> f32 {
        x
    }
}

impl <N: In> Connect<N> for InputNode<N> {
    fn connect(&mut self, weight: f32, node: InRef<N>) {
        self.next_nodes.push(node.clone())
    }
}