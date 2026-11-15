//! A minimal `.npy` 1.0 reader for the tests: the header's dtype and shape, and the body bytes. Independent of the
//! writer under test.

/// `\x93NUMPY` and version 1.0.
const PREAMBLE: &[u8] = b"\x93NUMPY\x01\x00";
const LENGTH_BYTES: usize = 2;
const HEADER_ALIGN: usize = 64;

pub struct Npy {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    pub body: Vec<u8>,
}

fn field<'a>(dict: &'a str, key: &str) -> &'a str {
    let start = dict
        .find(&format!("'{key}': "))
        .unwrap_or_else(|| panic!("{key} in {dict}"))
        + key.len()
        + 4;
    &dict[start..]
}

pub fn read(path: &std::path::Path) -> Npy {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(&raw[..PREAMBLE.len()], PREAMBLE, "magic and version");
    let at = PREAMBLE.len();
    let length = u16::from_le_bytes([raw[at], raw[at + 1]]) as usize;
    let data = at + LENGTH_BYTES + length;
    assert_eq!(data % HEADER_ALIGN, 0, "the data starts aligned");
    let dict = std::str::from_utf8(&raw[at + LENGTH_BYTES..data]).unwrap();
    assert!(dict.ends_with('\n'), "{dict:?}");
    let descr = field(dict, "descr");
    let descr = descr[1..descr[1..].find('\'').unwrap() + 1].to_string();
    let fortran_order = field(dict, "fortran_order").starts_with("True");
    let shape = field(dict, "shape");
    let shape = &shape[1..shape.find(')').unwrap()];
    let shape = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    Npy {
        descr,
        fortran_order,
        shape,
        body: raw[data..].to_vec(),
    }
}
