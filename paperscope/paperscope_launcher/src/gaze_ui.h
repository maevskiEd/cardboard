#ifndef PAPERSCOPE_LAUNCHER_SRC_GAZE_UI_H_
#define PAPERSCOPE_LAUNCHER_SRC_GAZE_UI_H_

#include <array>

#include <GLES2/gl2.h>
#include "util.h"

namespace cardboard_launcher {

/**
 * Draws the gaze reticle with its progress ring and the panel magnet. Both are
 * drawn procedurally in a shader, so no textures are needed.
 */
class GazeUi {
 public:
  GazeUi() = default;

  /**
   * Compiles the shader. Must be called on the rendering thread with a valid
   * GL context.
   */
  void Init();

  /**
   * Draws the reticle: a dot plus a ring filled clockwise up to |progress|.
   *
   * @param mvp Model view projection of a unit quad facing +Z.
   * @param color Reticle color.
   * @param progress Ring fill in [0, 1]; 0 hides the ring.
   */
  void DrawReticle(const cardboard_demo::Matrix4x4& mvp,
                   const std::array<float, 4>& color, float progress) const;

  /**
   * Draws the horseshoe magnet button.
   *
   * @param mvp Model view projection of a unit quad facing +Z.
   * @param color Button background color.
   */
  void DrawMagnet(const cardboard_demo::Matrix4x4& mvp,
                  const std::array<float, 4>& color) const;

 private:
  void DrawQuad(const cardboard_demo::Matrix4x4& mvp,
                const std::array<float, 4>& color, float icon,
                float progress) const;

  GLuint program_{0};
  GLuint position_param_{0};
  GLuint mvp_param_{0};
  GLuint color_param_{0};
  GLuint icon_param_{0};
  GLuint progress_param_{0};
};

}  // namespace cardboard_launcher

#endif  // PAPERSCOPE_LAUNCHER_SRC_GAZE_UI_H_
