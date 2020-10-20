use std::borrow::Cow;
use std::fmt::Display;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

const BASE_64_CONFIG: base64::Config = base64::URL_SAFE_NO_PAD;

#[derive(Clone)]
pub struct BlobKey<'a> {
    pub experiment_id: Cow<'a, str>,
    pub run: Cow<'a, str>,
    pub tag: Cow<'a, str>,
    pub step: i64,
    pub index: usize,
}

/// Helper to encode `BlobKey`s as tuple structs; pattern per @dtolnay:
/// <https://github.com/serde-rs/serde/issues/751>
#[derive(Serialize, Deserialize)]
struct WireBlobKey<'a>(&'a str, &'a str, &'a str, i64, usize);

#[derive(Debug)]
pub enum ParseBlobKeyError {
    BadBase64(base64::DecodeError),
    BadJson(serde_json::Error),
}

impl<'a> FromStr for BlobKey<'a> {
    type Err = ParseBlobKeyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let buf = base64::decode_config(s, BASE_64_CONFIG).map_err(ParseBlobKeyError::BadBase64)?;
        let wk: WireBlobKey = serde_json::from_slice(&buf).map_err(ParseBlobKeyError::BadJson)?;
        Ok(BlobKey::from_wire(&wk))
    }
}

impl<'a> Display for BlobKey<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use base64::display::Base64Display;
        let json = serde_json::to_string(&self.as_wire())
            .expect("wire blob keys should always be serializable");
        Base64Display::with_config(json.as_bytes(), BASE_64_CONFIG).fmt(f)
    }
}

impl<'a> BlobKey<'a> {
    fn as_wire(&self) -> WireBlobKey {
        WireBlobKey(
            &self.experiment_id,
            &self.run,
            &self.tag,
            self.step,
            self.index,
        )
    }

    fn from_wire(wk: &WireBlobKey) -> Self {
        BlobKey {
            experiment_id: Cow::Owned(wk.0.into()),
            run: Cow::Owned(wk.1.into()),
            tag: Cow::Owned(wk.2.into()),
            step: wk.3,
            index: wk.4,
        }
    }
}
