use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell
};

pub (mod) trait In {
    fn comp_in(inp: &Vec<f32>) -> f32;
    fn recieve(x: f32);
}

type InRef<N: In> = Rc<RefCell<Box<N>>>;