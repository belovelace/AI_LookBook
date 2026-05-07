import os
from src.pipeline.step1_base import run_step
from src.pipeline.step2_graphic import run_step as run_step2
from src.pipeline.step3_bottom import run_step as run_step3
from src.pipeline.step4_face import run_step as run_step4
from src.image_utils import load_and_normalize, save_png_bytes
from src.config import FIT_TOP, FIT_BTM


def orchestrate(out_dir: str, pose_id: int, face_path: str, body_path: str, background_path: str, top_path: str, bottom_path: str, shoe_path: str, model_cut_top_refs: list, model_cut_bottom_refs: list, pose_text: str, pose_view: str, face_meta: list = None, skip_steps: list = None, from_step: int = 1):
    skip_steps = skip_steps or []
    # prepare buffers
    face_buf = load_and_normalize(face_path, target_width=1600, crop_region='face') if face_path else None
    body_buf = load_and_normalize(body_path, target_width=1000, crop_region='center') if body_path else None
    background_buf = load_and_normalize(background_path, target_width=1000, crop_region='center') if background_path else None
    top_buf = load_and_normalize(top_path, target_width=1400, crop_region='top') if top_path else None
    bottom_buf = load_and_normalize(bottom_path, target_width=1100, crop_region='center') if bottom_path else None
    shoe_buf = load_and_normalize(shoe_path, target_width=1400, crop_region='shoe') if shoe_path else None
    wear_top_bufs = [load_and_normalize(p, target_width=800, crop_region='center') for p in model_cut_top_refs if p]
    wear_bottom_bufs = [load_and_normalize(p, target_width=800, crop_region='center') for p in model_cut_bottom_refs if p]

    current_step = from_step
    base_path = None
    top_path_out = None
    btm_path = None
    final_path = None

    # Step 1: base
    if current_step <= 1 and 1 not in skip_steps:
        parts = [b for b in (face_buf, body_buf, background_buf) if b]
        prompt = (
            "TASK: Generate a full-body fashion photograph. "
            "CRITICAL: The ENTIRE body must be visible head to toe — "
            "do NOT crop feet, ankles, or legs. Full body in frame. "

            f"POSE: {pose_text}. "
            "Use [Image #2] body proportions and pose ONLY — ignore the face in [Image #2]. "

            "FACE: Use [Image #1] face identity ONLY — "
            "preserve facial features, skin tone, eye shape. "
            "Ignore the face from [Image #2] completely. "

            "BACKGROUND: Pure white background with white floor. "
            "The floor must be BRIGHT WHITE — do NOT use gray, beige, or any colored floor. "
            "Ignore the floor color in [Image #2] — override it with pure white. "

            "GARMENT (text description only — no garment images at this stage): "
            "Top: black oversized round-neck short-sleeve T-shirt with pink flower graphic print on front center. "
            "Bottom: medium blue wash wide-leg denim pants, high waist, long length, wide leg opening. "

            "LABEL RULE: Do NOT show any clothing labels, tags, or brand text on the outside of garments. "
            "Internal labels are not visible in this view. "

            "Output: photorealistic editorial fashion photograph, aspect ratio 2:3."
        )
        img = run_step(parts, prompt)
        base_path = os.path.join(out_dir, f"outputs/steps/step1_{pose_id:02d}.png")
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        save_png_bytes(base_path, img)
        current_step = 2

    # Step 2: top graphic (use only top.jpg as graphic source)
    if current_step <= 2 and 2 not in skip_steps:
        parts = []
        with open(base_path, 'rb') as f:
            parts.append(f.read())
        if top_buf:
            parts.append(top_buf)
        # wear_top_bufs intentionally NOT used — top.jpg only

        prompt = (
            "TASK: GRAPHIC REPRODUCTION — apply the exact print from [Image #2] onto the T-shirt in [Image #1]. "
            "This is NOT a fashion photo generation task. This is a texture reproduction task. "

            "Reproduce EXACTLY from [Image #2]: "
            "- Pink flower/clover shape with rounded petals "
            "- Letters A, Zer, O inside the flower "
            "- L'infini script text below the flower "
            "- Black T-shirt background visible around the print "
            "Render as realistic screen-print: ink texture, slight cracking, fabric weave interaction. "
            "Deform print naturally with body pose and fabric folds. "

            "PRESERVE from [Image #1]: face, pose, background, pants — do NOT change anything except the T-shirt graphic area. "
            "FULL BODY must remain visible head to toe. "
            "Do NOT show any clothing labels or tags on the outside of the garment."
        )
        img2 = run_step2(parts, prompt)
        top_path_out = os.path.join(out_dir, f"outputs/steps/step2_{pose_id:02d}.png")
        os.makedirs(os.path.dirname(top_path_out), exist_ok=True)
        save_png_bytes(top_path_out, img2)
        current_step = 3

    # Step 3: bottom correction (view aware)
    if current_step <= 3 and 3 not in skip_steps:
        parts = []
        with open(top_path_out, 'rb') as f:
            parts.append(f.read())
        if bottom_buf:
            parts.append(bottom_buf)
        # include up to 2 worn-bottom references to convey fit/length/width/wash
        parts.extend(wear_bottom_bufs[:2])

        view_clause = ""
        if pose_view == 'front':
            view_clause = "View: FRONT. Do NOT show back waistband label; ensure no rear waistband logo is visible."
        elif pose_view == 'back':
            view_clause = "View: BACK. Show back details where appropriate; waistband label may be visible if consistent with garment reference."
        else:
            view_clause = "View: FRONT_45. No back waistband label visible; preserve 3/4 angle silhouette."

        prompt = (
            f"TASK: Bottom garment correction. {view_clause} "

            "Reference images: "
            "[Image #2] = bottom garment color and detail reference. "
            "[Image #3], [Image #4] = worn bottom reference — use ONLY for leg length, leg width, and wash color. "
            "Do NOT copy the person or face from [Image #3] or [Image #4]. "

            "CORRECT in [Image #1]: "
            "- Pants leg LENGTH: must reach ankle/floor level (long length) "
            "- Pants leg WIDTH: wide-leg silhouette, wide opening at hem "
            "- Denim WASH COLOR: medium blue, not dark navy, not light wash "
            "- Waist: high waist fit "

            "LABEL RULE: Do NOT show any internal labels, brand tags, or white label patches "
            "on the waistband exterior or anywhere visible on the garment. "

            "PRESERVE: T-shirt graphic, face, background floor color, pose. "
            "FULL BODY must remain visible head to toe."
        )
        img3 = run_step3(parts, prompt)
        btm_path = os.path.join(out_dir, f"outputs/steps/step3_{pose_id:02d}.png")
        os.makedirs(os.path.dirname(btm_path), exist_ok=True)
        save_png_bytes(btm_path, img3)
        current_step = 4

    # Step 4: face reinforcement (multi-ref)
    if current_step <= 4 and 4 not in skip_steps:
        parts = []
        with open(btm_path, 'rb') as f:
            parts.append(f.read())

        # multi-ref face load
        face_bufs_step4 = []
        if face_meta:
            for item in face_meta:
                try:
                    # support either (path, strategy) or dict entries
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        fp, strategy = item[0], item[1]
                    elif isinstance(item, dict) and 'path' in item:
                        fp, strategy = item['path'], item.get('strategy', {})
                    else:
                        continue

                    crop = None if (strategy.get('crop') == 'none') else strategy.get('crop')
                    tw = strategy.get('w') or 800
                    role = strategy.get('role') or 'FRONT_IDENTITY'
                    buf = load_and_normalize(fp, target_width=tw, crop_region=crop)
                    face_bufs_step4.append((buf, role))
                except Exception:
                    # ignore failed face ref loads
                    continue
        elif face_buf:
            face_bufs_step4.append((face_buf, 'FRONT_IDENTITY'))

        for buf, _role in face_bufs_step4:
            parts.append(buf)

        # build role descriptions for prompt (images are indexed starting at #2 for the first face ref)
        role_lines = ""
        for k, (_buf, role) in enumerate(face_bufs_step4, start=2):
            if role == "FRONT_IDENTITY":
                role_lines += f"[Image #{k}]: FACE IDENTITY — primary face structure, skin tone, eye shape.\n"
            elif role == "FACE_DETAIL":
                role_lines += f"[Image #{k}]: FACE DETAIL — eye/lip/skin texture close-up.\n"
            elif role == "PROFILE_SIDE":
                role_lines += f"[Image #{k}]: FACE PROFILE — side contour, nose shape, jawline.\n"
            else:
                role_lines += f"[Image #{k}]: {role}\n"

        prompt = (
            "TASK: FACE IDENTITY REPLACEMENT ONLY. "
            "Do NOT change: hair style, hair color, clothing, pose, background. "
            "Change ONLY: the face region. \n\n"

            f"Face reference images (all the SAME person, different angles):\n{role_lines}\n"

            "REPLACEMENT RULES: "
            "1. Use [Image #2] as primary identity anchor — match face structure and skin tone exactly. "
            "2. Use additional face refs for eye shape, lip color, nose contour detail. "
            "3. The face in the output MUST match the reference person — "
            "characteristics: monolid or single-lid eyes, wide forehead, full lips, fair skin. "
            "4. Do NOT retain the previous model's face (dark short hair model). "
            "5. The hair from [Image #1] source image must be PRESERVED — do not change hair. "
            "6. Output FULL BODY — head to toe visible."
        )
        img4 = run_step4(parts, prompt)
        final_path = os.path.join(out_dir, f"outputs/final/final_{pose_id:02d}.png")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        save_png_bytes(final_path, img4)
        current_step = 5

    # Step 5: shoe application (new)
    if shoe_buf and current_step <= 5 and 5 not in skip_steps:
        parts = []
        with open(final_path, 'rb') as f:
            parts.append(f.read())
        parts.append(shoe_buf)

        prompt = (
            "TASK: SHOE APPLICATION ONLY. "
            "Add the exact shoes from [Image #2] onto the model's feet in [Image #1]. "

            "SHOE REPRODUCTION RULES: "
            "Match exactly from [Image #2]: shoe color, shape, sole thickness, toe shape, material texture. "
            "Place shoes correctly on both feet with proper perspective and ground contact. "
            "Shoes must touch the floor — do NOT float. "

            "PRESERVE from [Image #1]: face, clothing, pose, background, floor color. "
            "Do NOT change anything except the foot/shoe area. "
            "FULL BODY must remain visible head to toe."
        )
        # reuse the face-step runner (step4) which is a general image-to-image call
        img5 = run_step4(parts, prompt)
        shoe_out_path = os.path.join(out_dir, f"outputs/final/final_{pose_id:02d}.png")
        os.makedirs(os.path.dirname(shoe_out_path), exist_ok=True)
        save_png_bytes(shoe_out_path, img5)
        final_path = shoe_out_path

    return final_path
