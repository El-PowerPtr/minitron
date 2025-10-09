use super::input::In;

pub trait Label<T>: In {
    fn label(&self) -> T;
}