#version 330

flat in vec3 v_color;
out vec4 frag_color;

// "Electron-cloud" aura around each reactive letter. Multi-shell radial
// falloff with subtle wisps so it reads as an organic cloud rather than a
// hard ring. Encodes the attractive-force range visually: falls to ~0 at the
// outer edge of the sprite, which the renderer sizes to the bond formation
// cutoff.
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = length(p);
    if (d > 1.0) discard;

    float core = exp(-d * d * 5.5) * 0.55;   // bright, tight center
    float mid  = exp(-d * d * 1.6) * 0.22;   // medium shell
    float far_ = exp(-d * 0.9)     * 0.10;   // wide diffuse halo
    float intensity = core + mid + far_;

    // Subtle radial wisps — gives it a textured, cloudlike feel.
    float wisp = 0.85 + 0.15 * sin(d * 9.0 + d * d * 3.5);
    intensity *= wisp;

    // Soften any sharp inner ring artifact at d→0.
    intensity *= smoothstep(1.0, 0.0, d);

    frag_color = vec4(v_color * intensity, intensity);
}
