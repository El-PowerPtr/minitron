use super::{
    output::Out,
    input::In,
};
use crate::conn::connection::{
    Connection, 
    ConnectsFrom,
    ConnectsTo,
};
use crate::{
    SharedRef,
};

pub struct InputNode<N: ConnectsFrom> {
    val: f32,
    next_nodes: Vec<SharedRef<Connection<Self,N>>>,
}

impl <N: ConnectsFrom>InputNode<N> {
    pub fn new() -> Self {
        InputNode{
            val: 0.0,
            next_nodes: vec![],
        }
    }
}

impl <N: ConnectsFrom> In for InputNode<N> {
    fn recieve(&mut self, x: f32) {
        self.val = x;
    }
}

impl <N: ConnectsFrom> ConnectsTo for InputNode<N>{
    type N = N;
    fn connect_to(&mut self, connection: SharedRef<Connection<Self, Self::N>>){
        self.next_nodes.push(connection);
    }
}

impl <N: ConnectsFrom> Out for InputNode<N> {
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
            (**n).borrow_mut().send_fronward(self.val);
        }
    }
}