#pragma once

#include "state/datastate.hpp"
#include "gl/GLLoader.hpp"

// Owns every GL object used to turn decoded video frames into upscaled
// pixels on screen. Three passes per frame:
//   A. decode:  YUV420P (or sws_scale-converted BGRA) source -> RGB, into an
//               FBO texture at the source frame's native resolution.
//   B. easu:    edge-adaptive upscale of that texture into an FBO texture
//               sized to the destination (letterboxed) rect.
//   C. rcas:    contrast-adaptive sharpen of the upscaled texture, drawn
//               straight to the backbuffer at the destination rect.
// One video output per process, so (like `window`/`renderer` used to be)
// this is deliberately a set of static globals rather than an instance.
class VideoRenderer
{
    public:
        // Call once, after Window::create_window() has made a GL context
        // current. Compiles shaders and allocates the quad VBO/VAO.
        static bool init();
        static void destroy();

        // Uploads a decoded frame to GL textures, converting on the CPU via
        // sws_scale first for any pixel format that isn't planar YUV420P.
        static bool upload_frame(AVFrame *frame, struct SwsContext **img_convert_ctx);

        // Runs the decode -> EASU -> RCAS pipeline for the most recently
        // uploaded frame, landing the final sharpened image in `rect`
        // within a `drawable_w`x`drawable_h` viewport. `flip_v` mirrors the
        // vertical flip SDL_RenderCopyEx used to apply for bottom-up frames.
        static void draw(const SDL_Rect &rect, int drawable_w, int drawable_h, bool flip_v);

        static void clear();
        static void present();

        // FSR RCAS sharpening strength in AMD's own "stops" convention
        // (FsrRcasCon): 0.0 = maximum sharpness, each +1.0 halves it
        // (internally, lobe *= exp2(-sharpness)).
        static void set_sharpness(float sharpness);

        // Toggles the EASU+RCAS passes on/off for A/B comparison. When off,
        // `draw()` falls back to a plain GL_LINEAR stretch of the decoded
        // frame into `rect` - the same quality the old SDL_Renderer path had.
        static void toggle_fsr();
        static bool fsr_enabled();

        // Cycles through AMD's published FSR1 quality presets (Native,
        // Ultra Quality 1.3x, Quality 1.5x, Balanced 1.7x, Performance 2.0x).
        // These control how much resolution EASU is asked to reconstruct:
        // the decoded frame is downsampled by the preset's ratio before
        // EASU upscales it back to display size, same as choosing a quality
        // mode in a game's FSR setting. Only affects anything while FSR is
        // enabled - see toggle_fsr().
        static void cycle_render_scale();
        static const char *render_scale_name();

    private:
        static bool ensure_yuv_textures(int width, int height);
        static bool ensure_rgba_texture(int width, int height);
        static bool ensure_decode_target(int width, int height);
        static bool ensure_upscale_target(int width, int height);
        static bool ensure_downscale_target(int width, int height);
        static void set_composite_quad(const SDL_Rect &rect, int drawable_w, int drawable_h);
};
