#version 330

flat in vec3 v_color;
out vec4 frag_color;

uniform float u_intensity;

void main() {
    frag_color = vec4(v_color * u_intensity, u_intensity);
}
