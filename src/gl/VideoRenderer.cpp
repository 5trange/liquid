#include "gl/VideoRenderer.hpp"
#include "gl/Shader.hpp"
#include "utils/Log.hpp"
#include <ios>
#include <vector>

namespace {

const char *vertex_src =
    "#version 330 core\n"
    "layout(location = 0) in vec2 aPos;\n"
    "layout(location = 1) in vec2 aTexCoord;\n"
    "out vec2 vTexCoord;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "    vTexCoord = aTexCoord;\n"
    "}\n";

// Decode pass: YUV420P -> RGB. uYuvMode: 0 = BT.601 limited range
// (default), 1 = BT.601 full range (JPEG), 2 = BT.709 limited range. uFlipV
// resolves bottom-up (negative linesize) frames here, once, so every later
// pass can assume a normal top-down image.
const char *fragment_src_decode_yuv =
    "#version 330 core\n"
    "in vec2 vTexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D uTexY;\n"
    "uniform sampler2D uTexU;\n"
    "uniform sampler2D uTexV;\n"
    "uniform int uYuvMode;\n"
    "uniform bool uFlipV;\n"
    "void main() {\n"
    "    vec2 uv = uFlipV ? vec2(vTexCoord.x, 1.0 - vTexCoord.y) : vTexCoord;\n"
    "    float y = texture(uTexY, uv).r;\n"
    "    float u = texture(uTexU, uv).r;\n"
    "    float v = texture(uTexV, uv).r;\n"
    "    vec3 rgb;\n"
    "    if (uYuvMode == 1) {\n"
    "        float d = u - 0.50196078;\n"
    "        float e = v - 0.50196078;\n"
    "        rgb = vec3(y + e * 1.402,\n"
    "                   y - d * 0.344136 - e * 0.714136,\n"
    "                   y + d * 1.772);\n"
    "    } else {\n"
    "        float c = (y - 0.06274510) * 1.164;\n"
    "        float d = u - 0.50196078;\n"
    "        float e = v - 0.50196078;\n"
    "        if (uYuvMode == 2) {\n"
    "            rgb = vec3(c + e * 1.793, c - d * 0.213 - e * 0.533, c + d * 2.112);\n"
    "        } else {\n"
    "            rgb = vec3(c + e * 1.596, c - d * 0.392 - e * 0.813, c + d * 2.017);\n"
    "        }\n"
    "    }\n"
    "    FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);\n"
    "}\n";

const char *fragment_src_decode_rgba =
    "#version 330 core\n"
    "in vec2 vTexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D uTex;\n"
    "uniform bool uFlipV;\n"
    "void main() {\n"
    "    vec2 uv = uFlipV ? vec2(vTexCoord.x, 1.0 - vTexCoord.y) : vTexCoord;\n"
    "    FragColor = texture(uTex, uv);\n"
    "}\n";

// Faithful port of AMD's ffx_fsr1.h FsrEasuF (non-packed 32-bit path),
// MIT licensed: https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h
// The only real change from AMD's source is mechanical: their version reads
// taps via Gather4 callbacks split across R/G/B channels (an optimization
// for their hardware gather trick); this reads each of the 12 taps with a
// plain texture() fetch instead. The direction/length estimation
// (FsrEasuSetF), the tap weighting (FsrEasuTapF, including its negative
// lobe - this is what gives it real sharpening character, unlike a plain
// clamped Lanczos), and the min4/max4 dering clamp are all unchanged.
const char *fragment_src_easu =
    "#version 330 core\n"
    "in vec2 vTexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D uTex;\n"
    "uniform vec2 uSrcSize;\n"
    "uniform vec2 uSrcTexelSize;\n"
    "\n"
    "float luma2(vec3 c) { return c.b * 0.5 + (c.r * 0.5 + c.g); }\n"
    "\n"
    "vec3 fetchTap(vec2 base, vec2 offset) {\n"
    "    return texture(uTex, (base + offset + 0.5) * uSrcTexelSize).rgb;\n"
    "}\n"
    "\n"
    "void easuSet(inout vec2 dir, inout float len, vec2 pp, bool biS, bool biT, bool biU, bool biV,\n"
    "             float lA, float lB, float lC, float lD, float lE) {\n"
    "    float w = 0.0;\n"
    "    if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);\n"
    "    if (biT) w = pp.x * (1.0 - pp.y);\n"
    "    if (biU) w = (1.0 - pp.x) * pp.y;\n"
    "    if (biV) w = pp.x * pp.y;\n"
    "    float dc = lD - lC;\n"
    "    float cb = lC - lB;\n"
    "    float lenX = 1.0 / max(abs(dc), abs(cb));\n"
    "    float dirX = lD - lB;\n"
    "    dir.x += dirX * w;\n"
    "    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);\n"
    "    lenX *= lenX;\n"
    "    len += lenX * w;\n"
    "    float ec = lE - lC;\n"
    "    float ca = lC - lA;\n"
    "    float lenY = 1.0 / max(abs(ec), abs(ca));\n"
    "    float dirY = lE - lA;\n"
    "    dir.y += dirY * w;\n"
    "    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);\n"
    "    lenY *= lenY;\n"
    "    len += lenY * w;\n"
    "}\n"
    "\n"
    "void easuTap(inout vec3 aC, inout float aW, vec2 offset, vec2 dir, vec2 len, float lob, float clp, vec3 c) {\n"
    "    vec2 v;\n"
    "    v.x = offset.x * dir.x + offset.y * dir.y;\n"
    "    v.y = offset.x * -dir.y + offset.y * dir.x;\n"
    "    v *= len;\n"
    "    float d2 = min(v.x * v.x + v.y * v.y, clp);\n"
    "    float wB = (2.0 / 5.0) * d2 - 1.0;\n"
    "    float wA = lob * d2 - 1.0;\n"
    "    wB *= wB;\n"
    "    wA *= wA;\n"
    "    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);\n"
    "    float w = wB * wA;\n"
    "    aC += c * w;\n"
    "    aW += w;\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    vec2 pp = vTexCoord * uSrcSize - 0.5;\n"
    "    vec2 base = floor(pp);\n"
    "    pp -= base;\n"
    "\n"
    "    vec3 b = fetchTap(base, vec2( 0.0, -1.0));\n"
    "    vec3 c = fetchTap(base, vec2( 1.0, -1.0));\n"
    "    vec3 e = fetchTap(base, vec2(-1.0,  0.0));\n"
    "    vec3 f = fetchTap(base, vec2( 0.0,  0.0));\n"
    "    vec3 g = fetchTap(base, vec2( 1.0,  0.0));\n"
    "    vec3 h = fetchTap(base, vec2( 2.0,  0.0));\n"
    "    vec3 i = fetchTap(base, vec2(-1.0,  1.0));\n"
    "    vec3 j = fetchTap(base, vec2( 0.0,  1.0));\n"
    "    vec3 k = fetchTap(base, vec2( 1.0,  1.0));\n"
    "    vec3 l = fetchTap(base, vec2( 2.0,  1.0));\n"
    "    vec3 n = fetchTap(base, vec2( 0.0,  2.0));\n"
    "    vec3 o = fetchTap(base, vec2( 1.0,  2.0));\n"
    "\n"
    "    float bL = luma2(b), cL = luma2(c), eL = luma2(e), fL = luma2(f);\n"
    "    float gL = luma2(g), hL = luma2(h), iL = luma2(i), jL = luma2(j);\n"
    "    float kL = luma2(k), lL = luma2(l), nL = luma2(n), oL = luma2(o);\n"
    "\n"
    "    vec2 dir = vec2(0.0);\n"
    "    float len = 0.0;\n"
    "    easuSet(dir, len, pp, true,  false, false, false, bL, eL, fL, gL, jL);\n"
    "    easuSet(dir, len, pp, false, true,  false, false, cL, fL, gL, hL, kL);\n"
    "    easuSet(dir, len, pp, false, false, true,  false, fL, iL, jL, kL, nL);\n"
    "    easuSet(dir, len, pp, false, false, false, true,  gL, jL, kL, lL, oL);\n"
    "\n"
    "    vec2 dir2 = dir * dir;\n"
    "    float dirR = dir2.x + dir2.y;\n"
    "    bool zro = dirR < (1.0 / 32768.0);\n"
    "    dirR = zro ? 1.0 : inversesqrt(dirR);\n"
    "    dir.x = zro ? 1.0 : dir.x;\n"
    "    dir *= dirR;\n"
    "\n"
    "    len = len * 0.5;\n"
    "    len *= len;\n"
    "\n"
    "    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));\n"
    "    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);\n"
    "\n"
    "    float lob = 0.5 + (0.25 - 0.04 - 0.5) * len;\n"
    "    float clp = 1.0 / lob;\n"
    "\n"
    "    vec3 min4 = min(min(f, g), min(j, k));\n"
    "    vec3 max4 = max(max(f, g), max(j, k));\n"
    "\n"
    "    vec3 aC = vec3(0.0);\n"
    "    float aW = 0.0;\n"
    "    easuTap(aC, aW, vec2( 0.0, -1.0) - pp, dir, len2, lob, clp, b);\n"
    "    easuTap(aC, aW, vec2( 1.0, -1.0) - pp, dir, len2, lob, clp, c);\n"
    "    easuTap(aC, aW, vec2(-1.0,  1.0) - pp, dir, len2, lob, clp, i);\n"
    "    easuTap(aC, aW, vec2( 0.0,  1.0) - pp, dir, len2, lob, clp, j);\n"
    "    easuTap(aC, aW, vec2( 0.0,  0.0) - pp, dir, len2, lob, clp, f);\n"
    "    easuTap(aC, aW, vec2(-1.0,  0.0) - pp, dir, len2, lob, clp, e);\n"
    "    easuTap(aC, aW, vec2( 1.0,  1.0) - pp, dir, len2, lob, clp, k);\n"
    "    easuTap(aC, aW, vec2( 2.0,  1.0) - pp, dir, len2, lob, clp, l);\n"
    "    easuTap(aC, aW, vec2( 2.0,  0.0) - pp, dir, len2, lob, clp, h);\n"
    "    easuTap(aC, aW, vec2( 1.0,  0.0) - pp, dir, len2, lob, clp, g);\n"
    "    easuTap(aC, aW, vec2( 1.0,  2.0) - pp, dir, len2, lob, clp, o);\n"
    "    easuTap(aC, aW, vec2( 0.0,  2.0) - pp, dir, len2, lob, clp, n);\n"
    "\n"
    "    vec3 result = min(max4, max(min4, aC / max(aW, 1e-6)));\n"
    "    FragColor = vec4(result, 1.0);\n"
    "}\n";

// Faithful port of AMD's ffx_fsr1.h FsrRcasF (non-packed 32-bit path), WITH
// FSR_RCAS_DENOISE enabled (always on, not behind a define here): it derives
// a per-pixel noise estimate from the same 5-tap neighborhood and multiplies
// the sharpen lobe by it, so compression/grain noise in near-flat regions
// doesn't get amplified into visible dithering the way plain sharpening
// would - real edges (where the 5 taps disagree in a structured, not
// noise-like, way) still get sharpened normally. uSharpness uses AMD's own
// convention from FsrRcasCon: "stops" where 0.0 = maximum sharpness and each
// +1.0 halves it (con.x = exp2(-sharpness), applied here directly).
const char *fragment_src_rcas =
    "#version 330 core\n"
    "in vec2 vTexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D uTex;\n"
    "uniform vec2 uTexelSize;\n"
    "uniform float uSharpness;\n"
    "float luma2(vec3 c) { return c.b * 0.5 + (c.r * 0.5 + c.g); }\n"
    "void main() {\n"
    "    vec3 b = texture(uTex, vTexCoord + vec2(0.0, -1.0) * uTexelSize).rgb;\n"
    "    vec3 d = texture(uTex, vTexCoord + vec2(-1.0, 0.0) * uTexelSize).rgb;\n"
    "    vec3 e = texture(uTex, vTexCoord).rgb;\n"
    "    vec3 f = texture(uTex, vTexCoord + vec2(1.0, 0.0) * uTexelSize).rgb;\n"
    "    vec3 h = texture(uTex, vTexCoord + vec2(0.0, 1.0) * uTexelSize).rgb;\n"
    "\n"
    "    vec3 mn4 = min(min(b, d), min(f, h));\n"
    "    vec3 mx4 = max(max(b, d), max(f, h));\n"
    "\n"
    "    float bL = luma2(b), dL = luma2(d), eL = luma2(e), fL = luma2(f), hL = luma2(h);\n"
    "    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;\n"
    "    float nzRange = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);\n"
    "    nz = clamp(abs(nz) / max(nzRange, 1e-6), 0.0, 1.0);\n"
    "    nz = -0.5 * nz + 1.0;\n"
    "\n"
    "    float peakCx = 1.0;\n"
    "    float peakCy = -4.0;\n"
    "    vec3 hitMin = min(mn4, e) / (4.0 * mx4 + 1e-6);\n"
    "    vec3 hitMax = (peakCx - max(mx4, e)) / (4.0 * mn4 + peakCy - 1e-6);\n"
    "    vec3 lobeRGB = max(-hitMin, hitMax);\n"
    "    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * exp2(-uSharpness) * nz;\n"
    "\n"
    "    float rcpL = 1.0 / (4.0 * lobe + 1.0);\n"
    "    vec3 result = (lobe * (b + d + f + h) + e) * rcpL;\n"
    "\n"
    "    FragColor = vec4(clamp(result, 0.0, 1.0), 1.0);\n"
    "}\n";

GLuint g_program_decode_yuv = 0;
GLuint g_program_decode_rgba = 0;
GLuint g_program_easu = 0;
GLuint g_program_rcas = 0;

GLint g_loc_decode_yuv_tex_y = -1, g_loc_decode_yuv_tex_u = -1, g_loc_decode_yuv_tex_v = -1;
GLint g_loc_decode_yuv_mode = -1, g_loc_decode_yuv_flip = -1;
GLint g_loc_decode_rgba_tex = -1, g_loc_decode_rgba_flip = -1;
GLint g_loc_easu_tex = -1, g_loc_easu_src_size = -1, g_loc_easu_texel_size = -1;
GLint g_loc_rcas_tex = -1, g_loc_rcas_texel_size = -1, g_loc_rcas_sharpness = -1;

GLuint g_vao_static = 0, g_vbo_static = 0;   // fixed fullscreen quad: decode + EASU passes
GLuint g_vao_composite = 0, g_vbo_composite = 0; // positioned into `rect`: final RCAS pass

GLuint g_tex_y = 0, g_tex_u = 0, g_tex_v = 0;
int g_y_w = 0, g_y_h = 0;
int g_uv_w = 0, g_uv_h = 0;

GLuint g_tex_rgba = 0;
int g_rgba_w = 0, g_rgba_h = 0;

bool g_using_yuv = true;
int g_yuv_mode = 0;

GLuint g_fbo_decode = 0, g_tex_decode = 0;
int g_decode_w = 0, g_decode_h = 0;

GLuint g_fbo_upscale = 0, g_tex_upscale = 0;
int g_upscale_w = 0, g_upscale_h = 0;

GLuint g_fbo_downscale = 0, g_tex_downscale = 0;
int g_downscale_w = 0, g_downscale_h = 0;

float g_sharpness = 0.2f; // AMD's "stops" convention: 0.0 = max sharpness
bool g_fsr_enabled = true;

// AMD's published FSR1 quality presets: the per-dimension ratio between the
// image EASU actually upscales from and the final display size. "Native"
// (1.0x, not an official FSR preset) feeds EASU the full decoded frame - the
// other four match AMD's real recommended ratios, deliberately throwing away
// resolution first so you can judge how much EASU can recover at each one,
// same as you'd compare them in a game's FSR quality setting.
struct RenderScalePreset { const char *name; float ratio; };
const RenderScalePreset kRenderScalePresets[] = {
    { "Native",        1.0f },
    { "Ultra Quality",  1.3f },
    { "Quality",        1.5f },
    { "Balanced",       1.7f },
    { "Performance",    2.0f },
};
const int kRenderScalePresetCount = sizeof(kRenderScalePresets) / sizeof(kRenderScalePresets[0]);
int g_scale_index = 0;

GLuint make_clamped_linear_texture()
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

} // namespace

bool VideoRenderer::init()
{
    g_program_decode_yuv = Shader::compile_program(vertex_src, fragment_src_decode_yuv);
    g_program_decode_rgba = Shader::compile_program(vertex_src, fragment_src_decode_rgba);
    g_program_easu = Shader::compile_program(vertex_src, fragment_src_easu);
    g_program_rcas = Shader::compile_program(vertex_src, fragment_src_rcas);
    if (!g_program_decode_yuv || !g_program_decode_rgba || !g_program_easu || !g_program_rcas)
        return false;

    g_loc_decode_yuv_tex_y = gl::GetUniformLocation(g_program_decode_yuv, "uTexY");
    g_loc_decode_yuv_tex_u = gl::GetUniformLocation(g_program_decode_yuv, "uTexU");
    g_loc_decode_yuv_tex_v = gl::GetUniformLocation(g_program_decode_yuv, "uTexV");
    g_loc_decode_yuv_mode = gl::GetUniformLocation(g_program_decode_yuv, "uYuvMode");
    g_loc_decode_yuv_flip = gl::GetUniformLocation(g_program_decode_yuv, "uFlipV");
    g_loc_decode_rgba_tex = gl::GetUniformLocation(g_program_decode_rgba, "uTex");
    g_loc_decode_rgba_flip = gl::GetUniformLocation(g_program_decode_rgba, "uFlipV");
    g_loc_easu_tex = gl::GetUniformLocation(g_program_easu, "uTex");
    g_loc_easu_src_size = gl::GetUniformLocation(g_program_easu, "uSrcSize");
    g_loc_easu_texel_size = gl::GetUniformLocation(g_program_easu, "uSrcTexelSize");
    g_loc_rcas_tex = gl::GetUniformLocation(g_program_rcas, "uTex");
    g_loc_rcas_texel_size = gl::GetUniformLocation(g_program_rcas, "uTexelSize");
    g_loc_rcas_sharpness = gl::GetUniformLocation(g_program_rcas, "uSharpness");

    gl::GenVertexArrays(1, &g_vao_static);
    gl::GenBuffers(1, &g_vbo_static);
    gl::BindVertexArray(g_vao_static);
    gl::BindBuffer(GL_ARRAY_BUFFER, g_vbo_static);
    // Texcoords are v-flipped relative to what you'd naively expect (top
    // vertex -> v=1) because this quad is used to render INTO an FBO
    // texture, not straight to the screen. NDC y=+1 rasterizes to the
    // framebuffer's last memory row, and a later texture(tex, v) call reads
    // v=0 from the FIRST memory row - so a plain top->v=0 quad comes out
    // upside down every time its output texture gets sampled again. Flipping
    // it here cancels that inversion so both the decode and EASU passes each
    // produce a right-side-up texture on their own, instead of only
    // "working" when both happen to run back to back and cancel out.
    static const float static_verts[16] = {
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
    };
    gl::BufferData(GL_ARRAY_BUFFER, sizeof(static_verts), static_verts, GL_STATIC_DRAW);
    gl::VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    gl::EnableVertexAttribArray(1);

    gl::GenVertexArrays(1, &g_vao_composite);
    gl::GenBuffers(1, &g_vbo_composite);
    gl::BindVertexArray(g_vao_composite);
    gl::BindBuffer(GL_ARRAY_BUFFER, g_vbo_composite);
    gl::BufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, NULL, GL_DYNAMIC_DRAW);
    gl::VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    gl::EnableVertexAttribArray(1);
    gl::BindVertexArray(0);

    gl::GenFramebuffers(1, &g_fbo_decode);
    gl::GenFramebuffers(1, &g_fbo_upscale);
    gl::GenFramebuffers(1, &g_fbo_downscale);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    return true;
}

void VideoRenderer::destroy()
{
    if (g_tex_y) glDeleteTextures(1, &g_tex_y);
    if (g_tex_u) glDeleteTextures(1, &g_tex_u);
    if (g_tex_v) glDeleteTextures(1, &g_tex_v);
    if (g_tex_rgba) glDeleteTextures(1, &g_tex_rgba);
    if (g_tex_decode) glDeleteTextures(1, &g_tex_decode);
    if (g_tex_upscale) glDeleteTextures(1, &g_tex_upscale);
    if (g_tex_downscale) glDeleteTextures(1, &g_tex_downscale);
    if (g_fbo_decode) gl::DeleteFramebuffers(1, &g_fbo_decode);
    if (g_fbo_upscale) gl::DeleteFramebuffers(1, &g_fbo_upscale);
    if (g_fbo_downscale) gl::DeleteFramebuffers(1, &g_fbo_downscale);
    if (g_vbo_static) gl::DeleteBuffers(1, &g_vbo_static);
    if (g_vao_static) gl::DeleteVertexArrays(1, &g_vao_static);
    if (g_vbo_composite) gl::DeleteBuffers(1, &g_vbo_composite);
    if (g_vao_composite) gl::DeleteVertexArrays(1, &g_vao_composite);
    if (g_program_decode_yuv) gl::DeleteProgram(g_program_decode_yuv);
    if (g_program_decode_rgba) gl::DeleteProgram(g_program_decode_rgba);
    if (g_program_easu) gl::DeleteProgram(g_program_easu);
    if (g_program_rcas) gl::DeleteProgram(g_program_rcas);

    g_tex_y = g_tex_u = g_tex_v = g_tex_rgba = 0;
    g_tex_decode = g_tex_upscale = g_tex_downscale = 0;
    g_fbo_decode = g_fbo_upscale = g_fbo_downscale = 0;
    g_vbo_static = g_vao_static = g_vbo_composite = g_vao_composite = 0;
    g_program_decode_yuv = g_program_decode_rgba = g_program_easu = g_program_rcas = 0;
    g_y_w = g_y_h = g_uv_w = g_uv_h = g_rgba_w = g_rgba_h = 0;
    g_decode_w = g_decode_h = g_upscale_w = g_upscale_h = g_downscale_w = g_downscale_h = 0;
}

bool VideoRenderer::ensure_yuv_textures(int width, int height)
{
    int uv_w = AV_CEIL_RSHIFT(width, 1);
    int uv_h = AV_CEIL_RSHIFT(height, 1);
    if (g_tex_y && g_y_w == width && g_y_h == height && g_uv_w == uv_w && g_uv_h == uv_h)
        return true;

    if (!g_tex_y) g_tex_y = make_clamped_linear_texture();
    if (!g_tex_u) g_tex_u = make_clamped_linear_texture();
    if (!g_tex_v) g_tex_v = make_clamped_linear_texture();

    glBindTexture(GL_TEXTURE_2D, g_tex_y);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, g_tex_u);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, uv_w, uv_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, g_tex_v);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, uv_w, uv_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    g_y_w = width;
    g_y_h = height;
    g_uv_w = uv_w;
    g_uv_h = uv_h;
    return true;
}

bool VideoRenderer::ensure_rgba_texture(int width, int height)
{
    if (g_tex_rgba && g_rgba_w == width && g_rgba_h == height)
        return true;

    if (!g_tex_rgba)
        g_tex_rgba = make_clamped_linear_texture();

    glBindTexture(GL_TEXTURE_2D, g_tex_rgba);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

    g_rgba_w = width;
    g_rgba_h = height;
    return true;
}

namespace {

bool ensure_render_target(GLuint fbo, GLuint &tex, int &cur_w, int &cur_h, int width, int height, const char *name)
{
    if (tex && cur_w == width && cur_h == height)
        return true;

    if (!tex)
        tex = make_clamped_linear_texture();

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    gl::BindFramebuffer(GL_FRAMEBUFFER, fbo);
    gl::FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    GLenum status = gl::CheckFramebufferStatus(GL_FRAMEBUFFER);
    gl::BindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        Log::error() << name << " framebuffer is incomplete (status 0x"
                     << std::hex << status << std::dec << ").";
        return false;
    }

    cur_w = width;
    cur_h = height;
    return true;
}

} // namespace

bool VideoRenderer::ensure_decode_target(int width, int height)
{
    return ensure_render_target(g_fbo_decode, g_tex_decode, g_decode_w, g_decode_h, width, height, "decode");
}

bool VideoRenderer::ensure_upscale_target(int width, int height)
{
    return ensure_render_target(g_fbo_upscale, g_tex_upscale, g_upscale_w, g_upscale_h, width, height, "upscale");
}

bool VideoRenderer::ensure_downscale_target(int width, int height)
{
    return ensure_render_target(g_fbo_downscale, g_tex_downscale, g_downscale_w, g_downscale_h, width, height, "downscale");
}

bool VideoRenderer::upload_frame(AVFrame *frame, struct SwsContext **img_convert_ctx)
{
    if (frame->format == AV_PIX_FMT_YUV420P) {
        if (!ensure_yuv_textures(frame->width, frame->height))
            return false;

        uint8_t *y_data = frame->data[0];
        uint8_t *u_data = frame->data[1];
        uint8_t *v_data = frame->data[2];
        int y_stride = frame->linesize[0];
        int u_stride = frame->linesize[1];
        int v_stride = frame->linesize[2];

        // Bottom-up frames (negative linesize) are re-pointed to their last
        // row with a positive stride, so uploads always walk top-to-bottom -
        // same normalization ffplay's upload_texture() did for SDL.
        if (y_stride < 0 && u_stride < 0 && v_stride < 0) {
            y_data += y_stride * (frame->height - 1);
            u_data += u_stride * (g_uv_h - 1);
            v_data += v_stride * (g_uv_h - 1);
            y_stride = -y_stride;
            u_stride = -u_stride;
            v_stride = -v_stride;
        } else if (y_stride < 0 || u_stride < 0 || v_stride < 0) {
            Log::error() << "Mixed negative and positive linesizes are not supported.";
            return false;
        }

        auto upload_plane = [](GLuint tex, uint8_t *data, int stride, int w, int h) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, data);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        };

        upload_plane(g_tex_y, y_data, y_stride, frame->width, frame->height);
        upload_plane(g_tex_u, u_data, u_stride, g_uv_w, g_uv_h);
        upload_plane(g_tex_v, v_data, v_stride, g_uv_w, g_uv_h);

        g_yuv_mode = 0;
        if (frame->color_range == AVCOL_RANGE_JPEG)
            g_yuv_mode = 1;
        else if (frame->colorspace == AVCOL_SPC_BT709)
            g_yuv_mode = 2;

        g_using_yuv = true;
        return true;
    }

    *img_convert_ctx = sws_getCachedContext(*img_convert_ctx,
        frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
        frame->width, frame->height, AV_PIX_FMT_BGRA,
        sws_flags, NULL, NULL, NULL);
    if (!*img_convert_ctx) {
        Log::error() << "Could not initialize conversion context!";
        return false;
    }

    if (!ensure_rgba_texture(frame->width, frame->height))
        return false;

    std::vector<uint8_t> buf((size_t)frame->width * frame->height * 4);
    uint8_t *dst_data[4] = { buf.data(), NULL, NULL, NULL };
    int dst_linesize[4] = { frame->width * 4, 0, 0, 0 };
    sws_scale(*img_convert_ctx, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);

    glBindTexture(GL_TEXTURE_2D, g_tex_rgba);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame->width, frame->height, GL_BGRA, GL_UNSIGNED_BYTE, buf.data());

    g_using_yuv = false;
    return true;
}

void VideoRenderer::set_composite_quad(const SDL_Rect &rect, int drawable_w, int drawable_h)
{
    float x0 = (float(rect.x) / drawable_w) * 2.0f - 1.0f;
    float x1 = (float(rect.x + rect.w) / drawable_w) * 2.0f - 1.0f;
    float y_top = 1.0f - (float(rect.y) / drawable_h) * 2.0f;
    float y_bot = 1.0f - (float(rect.y + rect.h) / drawable_h) * 2.0f;

    float verts[16] = {
        x0, y_top, 0.0f, 0.0f,
        x1, y_top, 1.0f, 0.0f,
        x0, y_bot, 0.0f, 1.0f,
        x1, y_bot, 1.0f, 1.0f,
    };

    gl::BindBuffer(GL_ARRAY_BUFFER, g_vbo_composite);
    gl::BufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
}

void VideoRenderer::draw(const SDL_Rect &rect, int drawable_w, int drawable_h, bool flip_v)
{
    if (rect.w <= 0 || rect.h <= 0)
        return;

    int src_w = g_using_yuv ? g_y_w : g_rgba_w;
    int src_h = g_using_yuv ? g_y_h : g_rgba_h;
    if (src_w <= 0 || src_h <= 0)
        return;

    if (!ensure_decode_target(src_w, src_h))
        return;
    if (g_fsr_enabled && !ensure_upscale_target(rect.w, rect.h))
        return;

    // Pass A: decode (YUV/BGRA -> RGB) into a native-resolution FBO.
    gl::BindFramebuffer(GL_FRAMEBUFFER, g_fbo_decode);
    glViewport(0, 0, src_w, src_h);
    gl::BindVertexArray(g_vao_static);
    if (g_using_yuv) {
        gl::UseProgram(g_program_decode_yuv);
        gl::ActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_tex_y);
        gl::ActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, g_tex_u);
        gl::ActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, g_tex_v);
        gl::Uniform1i(g_loc_decode_yuv_tex_y, 0);
        gl::Uniform1i(g_loc_decode_yuv_tex_u, 1);
        gl::Uniform1i(g_loc_decode_yuv_tex_v, 2);
        gl::Uniform1i(g_loc_decode_yuv_mode, g_yuv_mode);
        gl::Uniform1i(g_loc_decode_yuv_flip, flip_v ? 1 : 0);
    } else {
        gl::UseProgram(g_program_decode_rgba);
        gl::ActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_tex_rgba);
        gl::Uniform1i(g_loc_decode_rgba_tex, 0);
        gl::Uniform1i(g_loc_decode_rgba_flip, flip_v ? 1 : 0);
    }
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Pass B: EASU-style edge-adaptive upscale into a destination-resolution
    // FBO - skipped entirely when FSR is toggled off, so Pass C falls back
    // to sampling the decoded frame directly (plain GL_LINEAR stretch, same
    // quality the old SDL_Renderer path had) for a clean A/B comparison.
    GLuint composite_source = g_tex_decode;
    float composite_sharpness = 100.0f; // exp2(-100) ~= 0: lobe forced to 0, RCAS becomes an identity passthrough
    if (g_fsr_enabled) {
        GLuint easu_input_tex = g_tex_decode;
        int easu_src_w = src_w, easu_src_h = src_h;

        // Optional quality-preset pre-pass: deliberately throw away
        // resolution (a plain GL_LINEAR downsize, reusing the RCAS program
        // with sharpness forced off as a passthrough blit) before EASU, so
        // it has to reconstruct detail the same way it would for a game
        // rendered at less than native resolution. Native (ratio 1.0) skips
        // this and feeds EASU the full decoded frame.
        float scale_ratio = kRenderScalePresets[g_scale_index].ratio;
        if (scale_ratio > 1.0f) {
            int ds_w = (int)(src_w / scale_ratio + 0.5f);
            int ds_h = (int)(src_h / scale_ratio + 0.5f);
            if (ds_w < 1) ds_w = 1;
            if (ds_h < 1) ds_h = 1;
            if (!ensure_downscale_target(ds_w, ds_h))
                return;

            gl::BindFramebuffer(GL_FRAMEBUFFER, g_fbo_downscale);
            glViewport(0, 0, ds_w, ds_h);
            gl::UseProgram(g_program_rcas);
            gl::ActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g_tex_decode);
            gl::Uniform1i(g_loc_rcas_tex, 0);
            gl::Uniform2f(g_loc_rcas_texel_size, 1.0f / src_w, 1.0f / src_h);
            gl::Uniform1f(g_loc_rcas_sharpness, 100.0f);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            easu_input_tex = g_tex_downscale;
            easu_src_w = ds_w;
            easu_src_h = ds_h;
        }

        gl::BindFramebuffer(GL_FRAMEBUFFER, g_fbo_upscale);
        glViewport(0, 0, rect.w, rect.h);
        gl::UseProgram(g_program_easu);
        gl::ActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, easu_input_tex);
        gl::Uniform1i(g_loc_easu_tex, 0);
        gl::Uniform2f(g_loc_easu_src_size, (float)easu_src_w, (float)easu_src_h);
        gl::Uniform2f(g_loc_easu_texel_size, 1.0f / easu_src_w, 1.0f / easu_src_h);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        composite_source = g_tex_upscale;
        composite_sharpness = g_sharpness;
    }

    // Pass C: RCAS sharpen (or passthrough), drawn straight to the backbuffer at `rect`.
    gl::BindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, drawable_w, drawable_h);
    set_composite_quad(rect, drawable_w, drawable_h);
    gl::UseProgram(g_program_rcas);
    gl::ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, composite_source);
    gl::Uniform1i(g_loc_rcas_tex, 0);
    gl::Uniform2f(g_loc_rcas_texel_size, 1.0f / rect.w, 1.0f / rect.h);
    gl::Uniform1f(g_loc_rcas_sharpness, composite_sharpness);
    gl::BindVertexArray(g_vao_composite);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    gl::BindVertexArray(0);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
        Log::error() << "GL error 0x" << std::hex << err << std::dec << " during VideoRenderer::draw().";
}

void VideoRenderer::set_sharpness(float sharpness)
{
    // AMD's "stops" convention (FsrRcasCon): 0.0 = maximum sharpness, each
    // +1.0 halves it. Only clamped from below - negative stops would push
    // the lobe past the stability limit ffx_fsr1.h derives it from.
    g_sharpness = sharpness < 0.0f ? 0.0f : sharpness;
}

void VideoRenderer::toggle_fsr()
{
    g_fsr_enabled = !g_fsr_enabled;
}

bool VideoRenderer::fsr_enabled()
{
    return g_fsr_enabled;
}

void VideoRenderer::cycle_render_scale()
{
    g_scale_index = (g_scale_index + 1) % kRenderScalePresetCount;
}

const char *VideoRenderer::render_scale_name()
{
    return kRenderScalePresets[g_scale_index].name;
}

void VideoRenderer::clear()
{
    gl::BindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void VideoRenderer::present()
{
    SDL_GL_SwapWindow(window);
}
