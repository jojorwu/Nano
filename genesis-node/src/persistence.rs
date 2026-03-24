use genesis_core::BakedModel;
use crate::RuntimeError;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub struct PersistenceManager;

impl PersistenceManager {
    pub fn save(model: &BakedModel, path: &str) -> Result<(), RuntimeError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, model).map_err(|e| RuntimeError::Serialization(e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<BakedModel, RuntimeError> {
        let file = File::open(path).map_err(|e| RuntimeError::IoWithPath(path.to_string(), e))?;
        let reader = BufReader::new(file);
        let model: BakedModel = bincode::deserialize_from(reader).map_err(|e| RuntimeError::SerializationContext(format!("deserializing model from {}", path), e))?;
        Ok(model)
    }
}
