use cuda_device::{kernel, thread};

fn neg_inf() -> f32 {
    -3.4028235e38_f32
}

fn floor_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if (truncated as f32) > value {
        truncated - 1
    } else {
        truncated
    }
}

fn pow2_i32(exp: i32) -> f32 {
    if exp < -126 {
        0.0
    } else if exp > 127 {
        3.4028235e38_f32
    } else {
        f32::from_bits(((exp + 127) as u32) << 23)
    }
}

fn exp_f32(value: f32) -> f32 {
    if value <= -87.0 {
        return 0.0;
    }
    if value >= 88.0 {
        return 3.4028235e38_f32;
    }

    let k = floor_i32(value * 1.442_695_1);
    let r = value - (k as f32) * 0.693_147_2;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r3 * r3;
    let r7 = r6 * r;
    let poly = 1.0
        + r
        + 0.5 * r2
        + 0.166_666_67 * r3
        + 0.041_666_668 * r4
        + 0.008_333_333 * r5
        + 0.001_388_888_9 * r6
        + 0.000_198_412_7 * r7;

    poly * pow2_i32(k)
}

fn tanh_f32(value: f32) -> f32 {
    if value >= 9.0 {
        1.0
    } else if value <= -9.0 {
        -1.0
    } else {
        let e = exp_f32(2.0 * value);
        (e - 1.0) / (e + 1.0)
    }
}

#[kernel]
pub unsafe fn scaled_softmax_oxide(
    x: *mut f32,
    len: usize,
    n_rows: i32,
    n_cols: i32,
    scale: f32,
    softcap: f32,
    causal: i32,
) {
    let row = thread::index_1d().get();
    let n_rows = n_rows as usize;
    let n_cols = n_cols as usize;
    if row >= n_rows {
        return;
    }
    let base = row * n_cols;
    if base + n_cols > len {
        return;
    }

    let mut row_max = neg_inf();
    let mut col = 0usize;
    while col < n_cols {
        let ptr = unsafe { x.add(base + col) };
        let mut value = unsafe { *ptr } * scale;
        if softcap > 0.0 {
            value = softcap * tanh_f32(value / softcap);
        }
        if causal != 0 && col > row {
            value = neg_inf();
        }
        unsafe {
            *ptr = value;
        }
        if value > row_max {
            row_max = value;
        }
        col += 1;
    }

    let mut row_sum = 0.0_f32;
    col = 0;
    while col < n_cols {
        let ptr = unsafe { x.add(base + col) };
        let value = exp_f32(unsafe { *ptr } - row_max);
        unsafe {
            *ptr = value;
        }
        row_sum += value;
        col += 1;
    }

    let inv_sum = 1.0 / row_sum;
    col = 0;
    while col < n_cols {
        let ptr = unsafe { x.add(base + col) };
        unsafe {
            *ptr *= inv_sum;
        }
        col += 1;
    }
}
