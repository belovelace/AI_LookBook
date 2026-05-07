#!/usr/bin/env python3
import os
import sys
import argparse
from src.config import GEMINI_API_KEY, validate_config, MODEL, OUTPUT_DIR
from src.image_utils import load_and_normalize, save_png_bytes
from src.gemini_client import generate
from src.logging_utils import make_output_dir, append_run_log, print_summary
from src.pipeline.orchestrator import orchestrate
import time
import glob

POSES = [
    "standing upright, arms relaxed at sides, looking straight into camera",
    "walking forward naturally, slight stride, one foot in front of the other",
    "standing with one hand in pocket, other arm relaxed, slight weight shift to one leg",
    "turning slightly to the side, looking back over shoulder toward camera",
    "standing with both hands in pockets, chin slightly down, relaxed expression",
    "walking at a slight angle, arms swinging naturally, looking ahead",
    "standing with arms crossed loosely at waist, looking straight at camera",
    "one foot slightly forward, weight on back leg, arms relaxed, direct gaze",
    "standing with hands clasped in front, slight tilt of head, soft expression",
    "mid-stride walk, turned 3/4 angle toward camera, natural arm movement",
]
FIT_TOP  = "검정색 반팔 티셔츠. 앞면 중앙에 핑크색 꽃 모양 그래픽이 크게 프린트되어 있고, 꽃 안에 'A', 'Zer', 'O' 텍스트와 'L'infini' 글씨가 적혀있다. 넥라인 안쪽에 'AZERO L'INFINI' 라벨이 있다. 라운드넥, 짧은 소매, 밑단이 살짝 둥글게 커팅된 오버사이즈 핏이다."
FIT_BTM  = "미디엄 블루 워싱의 와이드 데님 팬츠. 하이웨이스트이며 다리통이 넓고 기장이 길다. 앞면에 5포켓 디테일과 실버 버튼, 골드 스티칭이 있다. 'AZERO L'INFINI' 라벨이 허리 안쪽에 있다."
MATERIAL = "상의: 면 소재, 매트한 질감 / 하의: 데님 소재, 약간의 워싱 처리"


def build_prompt(pose, face_refs=None, body_ref="assets/model_body.png", outfit_refs=None, image_index_map=None, image_role_map=None, shoe_index=None, primary_face_idx=None, pose_view="front"):
    if image_role_map is None:
        image_role_map = {}
    """Build a concise, view-aware prompt.
    pose_view: 'front', 'back', or 'front_45'
    """
    if face_refs is None:
        face_refs = ["assets/face.jpg"]
    if outfit_refs is None:
        outfit_refs = []
    if image_index_map is None:
        image_index_map = {}

    mapping_lines = ""
    for fp, idx in sorted(image_index_map.items(), key=lambda x: x[1]):
        mapping_lines += f"- [Image #{idx}]: {fp}\n"

    # face role descriptions
    ROLE_DESC = {
        "FRONT_IDENTITY": "정면 얼굴 — identity 기준 (얼굴 구조, 피부톤, 눈 형태)",
        "FACE_DETAIL":    "얼굴 근접 크롭 — 눈/코/입 디테일, 피부 텍스처 기준",
        "PROFILE_SIDE":   "측면 얼굴 — 윤곽선, 코 형태, 턱선 기준",
    }

    face_list_text = ""
    if face_refs:
        for fp in face_refs:
            img_idx = image_index_map.get(fp, "?")
            role = image_role_map.get(fp, "FACE_REFERENCE")
            desc = ROLE_DESC.get(role, "얼굴 참조")
            face_list_text += f"- [Image #{img_idx}]: {desc}\n"

    outfit_list_text = ""
    if outfit_refs:
        for i, fp in enumerate(outfit_refs, start=1):
            outfit_list_text += f"- 의상 참조 {i}: {fp} (착용 참조 — fabric/fit only)\n"

    # Shoe instruction (short)
    shoe_instruction = ""
    if shoe_index is not None:
        shoe_instruction = (
            f"Use shoes from [Image #{shoe_index}] exactly — match color, shape, sole and scale.\n"
        )

    # Strong but concise identity instruction
    face_instruction = "Use the face from [Image #1] exactly — preserve facial features, skin tone and expression. Do NOT alter identity.\n"

    # Short wearing refs guidance
    wearing_refs_clause = "Use wearing-reference images ONLY for fabric behavior, fit and wrinkles; do NOT copy persons or faces.\n"

    # View-specific short clause
    view_clause = ""
    if pose_view == "front":
        view_clause = "View: FRONT. Do NOT show a back waistband label; ensure no rear waistband logo is visible.\n"
    elif pose_view == "back":
        view_clause = "View: BACK. Show back details where appropriate; waistband label may be visible if consistent with garment reference.\n"
    elif pose_view == "front_45":
        view_clause = "View: FRONT_45. No back waistband label visible; preserve 3/4 angle silhouette.\n"

    core = (
        f"{view_clause}"
        f"{face_instruction}"
        f"Use body proportions and pose framing from {body_ref} (body/pose only).\n"
        f"Place model into background matching lighting and floor from assets/background.jpg.\n"
        f"Dress in the exact garments: top from {outfit_refs[0] if outfit_refs else 'top ref'} and bottom from {outfit_refs[1] if len(outfit_refs) > 1 else 'bottom ref'}.\n"
        f"{wearing_refs_clause}"
        "Do NOT hallucinate logos, text, additional people, or accessories not present in the references.\n"
        "Photorealistic editorial fashion photograph, aspect ratio 2:3, PNG."
    )

    # minimal mapping + details appended
    return (
        "참고 이미지 목록:\n"
        f"{mapping_lines}"
        f"{face_list_text}"
        f"- 전신 모델 이미지: {body_ref} (모델의 체형/비율/포즈 참조)\n"
        f"{outfit_list_text}"
        f"{core}\n"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", help="Comma-separated pose list", default=None)
    p.add_argument("--pose-view", help="Pose view: front|back|front_45", choices=["front","back","front_45"], default="front")
    p.add_argument("--output", help="Output base dir", default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    validate_config()

    poses = POSES
    if args.poses:
        poses = [s.strip() for s in args.poses.split(",") if s.strip()]

    # locate base images
    face_files = []
    if os.path.exists("assets/face.jpg"):
        face_files.append("assets/face.jpg")
    face_files.extend(sorted(glob.glob("assets/face/*")))
    seen = set()
    face_files = [x for x in face_files if not (x in seen or seen.add(x))]

    body_path = "assets/model_body.png" if os.path.exists("assets/model_body.png") else None
    if not body_path:
        print("Missing assets/model_body.png (full body reference)")
        sys.exit(1)

    # collect outfit refs recursively
    outfit_files = []
    outfit_dir = os.path.join("outfits", "model_cut")
    if os.path.exists(outfit_dir):
        for root, _, files in os.walk(outfit_dir):
            for fname in sorted(files):
                fp = os.path.join(root, fname)
                if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tiff"}:
                    outfit_files.append(fp)

    background_path = "assets/background.jpg" if os.path.exists("assets/background.jpg") else None

    out_dir = make_output_dir(args.output)
    total = len(poses)
    success = 0

    for i, pose in enumerate(poses, start=1):
        print(f"[{i}/{total}] 생성 중... {pose}")

        # pick primaries to avoid dilution
        # face loading strategy: explicit roles and sizes to preserve identity
        FACE_LOAD_STRATEGY = {
            "face.jpg":               {"crop": "face_upper",  "w": 1200, "role": "FRONT_IDENTITY"},
            "face_crop.png":          {"crop": "none",         "w": 1400, "role": "FACE_DETAIL"},
            "Profile_Side_Angle.png": {"crop": "face_upper",  "w": 1000, "role": "PROFILE_SIDE"},
        }

        face_meta = []
        for fname, strategy in FACE_LOAD_STRATEGY.items():
            candidates = [
                os.path.join("assets", "face", fname),
                os.path.join("assets", fname),
            ]
            for fp in candidates:
                if os.path.exists(fp):
                    face_meta.append((fp, strategy))
                    break

        # primary_faces still kept for backward compatibility but we use face_meta for loading
        primary_faces = [f for f in face_files]  # fallback
        image_role_map = {}
        # prioritize explicit outfit files (top/bottom/shose) if present, then model_cut refs
        top_path = os.path.join("outfits", "top.jpg")
        bottom_path = os.path.join("outfits", "bottom.jpg")
        shose_path = os.path.join("outfits", "shose.jpg")

        primary_outfits = []
        if os.path.exists(top_path):
            primary_outfits.append(top_path)
        if os.path.exists(bottom_path):
            primary_outfits.append(bottom_path)
        if os.path.exists(shose_path):
            primary_outfits.append(shose_path)
        # then add any model_cut refs (already collected into outfit_files)
        for fp in outfit_files:
            if fp not in primary_outfits:
                primary_outfits.append(fp)

        # determine shoe candidate
        send_order = []
        # faces first to bias identity
        send_order.extend(primary_faces)
        # send explicit top/bottom/shose first to ensure primary garments are present
        send_order.extend([p for p in primary_outfits if os.path.basename(p).lower() in (os.path.basename(top_path).lower(), os.path.basename(bottom_path).lower(), os.path.basename(shose_path).lower()) if p])
        # then send other primary outfits (model_cut refs)
        send_order.extend([p for p in primary_outfits if p not in send_order])
        if body_path:
            send_order.append(body_path)
        if background_path:
            send_order.append(background_path)

        shoe_fp = None
        for candidate in send_order:
            name = os.path.basename(candidate).lower()
            if any(k in name for k in ("shoe","sneaker","heel","loafer","boot")):
                shoe_fp = candidate
                break
        if shoe_fp is None and primary_outfits:
            shoe_fp = primary_outfits[0]

        # safe loader helper
        def try_load(fp, target_w=800, crop_region=None):
            try:
                return load_and_normalize(fp, target_width=target_w, crop_region=crop_region)
            except Exception as e:
                print(f"[WARN] Skipping file ({fp}): {e}")
                append_run_log(out_dir, "SKIPPED", os.path.basename(fp), str(e))
                return None

        # load with bias: face high-res duplicate + shoe high-res last
        images = []
        image_index_map = {}
        sent_files = []
        idx = 1

        # load faces according to face_meta strategy (preserve roles)
        primary_face_bytes = None
        for j, (fp, strategy) in enumerate(face_meta):
            crop = strategy.get("crop")
            tw   = strategy.get("w")
            role = strategy.get("role")

            crop_region = None if crop == "none" else crop

            buf = try_load(fp, target_w=tw, crop_region=crop_region)
            if buf is not None:
                images.append(buf)
                image_index_map[fp] = idx
                image_role_map[fp] = role
                sent_files.append(fp)
                if j == 0:
                    primary_face_bytes = buf
                idx += 1

        # load primary outfits with region-aware crop
        for fp in primary_outfits:
            if not fp:
                continue
            lower = os.path.basename(fp).lower()
            # prioritize explicit top/bottom/shose cropping rules
            if lower == os.path.basename(shose_path).lower() or any(k in lower for k in ("shoe","sneaker","boot","heel","loafer")):
                buf = try_load(fp, target_w=1400, crop_region='shoe')
            elif lower == os.path.basename(top_path).lower() or any(k in lower for k in ("top","shirt","tee","tshirt","blouse")):
                buf = try_load(fp, target_w=1100, crop_region='top')
            elif lower == os.path.basename(bottom_path).lower() or any(k in lower for k in ("jean","denim","pant","pants","trouser")):
                buf = try_load(fp, target_w=1100, crop_region='center')
            else:
                # model_cut refs — prefer center crop to capture fabric texture
                buf = try_load(fp, target_w=1000, crop_region='center')
            if buf is not None:
                images.append(buf)
                image_index_map[fp] = idx
                sent_files.append(fp)
                idx += 1

        # ensure body and background are included after garment refs
        if body_path:
            buf = try_load(body_path, target_w=1000, crop_region='center')
            if buf is not None:
                images.append(buf)
                image_index_map[body_path] = idx
                sent_files.append(body_path)
                idx += 1
        if background_path:
            buf = try_load(background_path, target_w=1000, crop_region='center')
            if buf is not None:
                images.append(buf)
                image_index_map[background_path] = idx
                sent_files.append(background_path)
                idx += 1

        # also ensure explicit top/bottom/shose are included if they weren't in primary_outfits loop
        for explicit in (top_path, bottom_path, shose_path):
            if explicit and explicit not in sent_files and os.path.exists(explicit):
                buf = try_load(explicit, target_w=1100, crop_region='center')
                if buf is not None:
                    images.append(buf)
                    image_index_map[explicit] = idx
                    sent_files.append(explicit)
                    idx += 1

        # append high-res shoe at the end to bias footwear — duplicate it twice for extra bias
        if shoe_fp:
            dup_count = 2
            shoe_index = None
            for n in range(dup_count):
                # slightly vary target width for duplicates to avoid exact-duplicate filtering
                tw = 1600 - (n * 100)
                buf = try_load(shoe_fp, target_w=tw, crop_region='shoe')
                if buf is not None:
                    images.append(buf)
                    key = shoe_fp + f"::highres{n+1}"
                    image_index_map[key] = idx
                    sent_files.append(key)
                    # set shoe_index to the first duplicate's index (lowest priority among duplicates is last one)
                    if n == 0:
                        shoe_index = idx
                    idx += 1
        else:
            shoe_index = None

        if not images:
            append_run_log(out_dir, "FAILED", f"pose_{i:02d}.png", "no readable reference images")
            print(f"[FAILED] no readable reference images to send")
            continue

        sent_outfit_refs = [f for f in primary_outfits if f in sent_files or f + "::highres" in sent_files]
        sent_face_refs = [fp for fp, _ in face_meta if fp in sent_files]

        # prefer the face image index explicitly if available
        primary_face_idx = None
        if sent_face_refs:
            # map first sent face ref to its image index
            fp = sent_face_refs[0]
            primary_face_idx = image_index_map.get(fp) or image_index_map.get(fp + "::highres")

        prompt = build_prompt(
                pose,
                face_refs=sent_face_refs,
                body_ref=body_path,
                outfit_refs=sent_outfit_refs,
                image_index_map=image_index_map,
                image_role_map=image_role_map,
                shoe_index=shoe_index,
                primary_face_idx=primary_face_idx,
                pose_view=args.pose_view,
            )

        # stronger top-print instruction if a top ref exists
        top_ref = next((p for p in sent_outfit_refs if 'top' in os.path.basename(p).lower() or 'shirt' in os.path.basename(p).lower()), None)
        if top_ref:
            top_idx = image_index_map.get(top_ref) or image_index_map.get(top_ref + "::highres")
            if top_idx:
                prompt += (
                    f"\nTop instruction — CRITICAL: Reproduce the EXACT graphic from [Image #{top_idx}] on the T-shirt. "
                    "Preserve flower shape, scale, placement, pink tone, distressed edges, typography layout and spacing. "
                    "Render as realistic SCREEN-PRINT onto the fabric (NOT pasted or vector); preserve ink texture, slight cracking and unevenness. "
                    "Ensure print deforms naturally with body curvature (wrinkles/tension) using wearing refs; do NOT alter, simplify, re-center, invent, or replace the graphic."
                )

        # debug output
        print("[DEBUG] image_index_map:", image_index_map)
        print("[DEBUG] sent_files:", sent_files)
        print("[DEBUG] shoe_index:", shoe_index)
        try:
            print("[DEBUG] prompt:\n" + (prompt[:4000] + "..." if len(prompt) > 4000 else prompt))
        except UnicodeEncodeError:
            safe = (prompt[:4000] + "...") if len(prompt) > 4000 else prompt
            enc = sys.stdout.encoding or 'utf-8'
            safe = safe.encode(enc, errors='replace').decode(enc)
            print("[DEBUG] prompt (safe):\n" + safe)

        # call orchestrator (multi-stage pipeline)
        try:
            final_out = orchestrate(
                out_dir,
                i,
                sent_face_refs[0] if sent_face_refs else None,
                body_path,
                background_path,
                top_path if os.path.exists(top_path) else None,
                bottom_path if os.path.exists(bottom_path) else None,
                shoe_fp,
                [p for p in outfit_files if os.path.dirname(p).lower().endswith(os.path.join('model_cut','model_top'))],
                [p for p in outfit_files if os.path.dirname(p).lower().endswith(os.path.join('model_cut','model_bottom'))],
                pose,
                args.pose_view,
                face_meta=face_meta,
            )
            append_run_log(out_dir, "SUCCESS", os.path.basename(final_out), pose)
            print(f"[SUCCESS] {final_out}")
            success += 1
        except Exception as e:
            append_run_log(out_dir, "FAILED", f"pose_{i:02d}.png", f"{e}")
            print(f"[FAILED] {e}")

    print_summary(out_dir, total, success)


if __name__ == "__main__":
    main()
