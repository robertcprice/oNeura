//! Vector and matrix math utilities for the software rasterizer.

pub type V3 = [f32; 3];
pub type V4 = [f32; 4];
pub type M4 = [[f32; 4]; 4];

pub fn v3(x: f32, y: f32, z: f32) -> V3 { [x, y, z] }
pub fn add3(a: V3, b: V3) -> V3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
pub fn sub3(a: V3, b: V3) -> V3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
pub fn scale3(a: V3, s: f32) -> V3 { [a[0]*s, a[1]*s, a[2]*s] }
pub fn dot3(a: V3, b: V3) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
pub fn cross3(a: V3, b: V3) -> V3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
pub fn len3(a: V3) -> f32 { dot3(a, a).sqrt() }
pub fn normalize3(a: V3) -> V3 {
    let l = len3(a);
    if l < 1e-9 { [0.0, 1.0, 0.0] } else { scale3(a, 1.0 / l) }
}
pub fn lerp3(a: V3, b: V3, t: f32) -> V3 {
    [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]
}

pub fn mat4_mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 {
        r[i][j] += a[i][k] * b[k][j];
    }}}
    r
}

pub fn transform4(m: &M4, v: V4) -> V4 {
    let mut r = [0.0f32; 4];
    for i in 0..4 { for j in 0..4 { r[i] += m[i][j] * v[j]; } }
    r
}

pub fn look_at(eye: V3, target: V3, up: V3) -> M4 {
    let f = normalize3(sub3(target, eye));
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);
    [
        [s[0], s[1], s[2], -dot3(s, eye)],
        [u[0], u[1], u[2], -dot3(u, eye)],
        [-f[0], -f[1], -f[2], dot3(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> M4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ]
}
