use crate::{
    MultiRef,
    nodes::{
        input::In,
        input_node::InputNode,
        output::Out,
        learn::*,
    },
};
use super::connect::Connect;

pub struct Connection <F: Connect + Out,T: In> {
    pub from: MultiRef<F>,
    pub weight: f32,
    err: f32,
    pub to: MultiRef<T>,
}

impl <F: Connect + Out, T: In + BackProp> Learn for Connection <F, T>{
    fn learn(&mut self){
        self.weight -= self.err;
    }
    fn get_feedback(&mut self, feedback: f32){
        self.err *= feedback;
    }
}

impl <T: In> BackProp for Connection <InputNode<T>, T>{
    fn send_feedback(&mut self){
        //nothing
    }
}

impl <F: Connect + Out + Learn, T: In> BackProp for Connection<F,T>{
    fn send_feedback(&mut self){
        self.from.borrow_mut().get_feedback(self.err * self.weight)
    }
}

impl <F: Connect + Out, T: In> Connection<F,T>  {
    pub fn send_fronward(&mut self, out: f32){
        self.err = out;
        self.to.borrow_mut().recieve(out);
    }
}