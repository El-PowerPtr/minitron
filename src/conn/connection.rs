use crate::{
    MultiRef,
    SharedRef,
    nodes::{
        input::In,
        input_node::InputNode,
        output::Out,
        learn::*,
    },
};

pub trait Connectable{
    fn add_connection<F: Out, T: In>(&mut self, connection: SharedRef<Connection<F,T>>);
}

pub struct Connection <F: Out,T: In> {
    pub from: MultiRef<F>,
    pub weight: f32,
    err: f32,
    pub to: MultiRef<T>,
}

impl <F: Out,T: In>Connection<F,T>{
    pub fn new(from: MultiRef<F>, to: MultiRef<T>, weight: f32) -> Self{
        Connection {
            from,
            weight,
            err: 0.0,
            to,
        }
    }
}

impl <F: Out, T: In + BackProp> Learn for Connection <F, T>{
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

impl <F:  Out + Learn, T: In> BackProp for Connection<F,T>{
    fn send_feedback(&mut self){
        self.from.borrow_mut().get_feedback(self.err * self.weight)
    }
}

impl <F:  Out, T: In> Connection<F,T>  {
    pub fn send_fronward(&mut self, out: f32){
        self.err = out;
        self.to.borrow_mut().recieve(out);
    }
}