# ============================================================
# ✅ app.py — Nutrition Chatbot (Ollama) + User Profile + RAG
# ============================================================
# الفكرة:
# 1) المستخدم يعبّي بياناته (Profile)
# 2) RAG يبحث داخل ملف FoodData Central JSON ويسترجع أفضل نتائج
# 3) نرسل (السؤال + البيانات المسترجعة + البروفايل) إلى Ollama
#
# ✅ المتطلبات (ثبّتيها مرة واحدة على جهازك):
# pip install gradio ollama sentence-transformers faiss-cpu numpy
#
# ✅ لازم يكون ملف JSON في نفس مجلد app.py بهذا الاسم:
# FoodData_Central_foundation_food_json_2025-12-18.json
# ============================================================

import os
import json
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama


# ============================================================
# 1) إعدادات عامة
# ============================================================
MODEL_NAME = "nutrition-phi3-healthcoach"
JSON_PATH = "FoodData_Central_foundation_food_json_2025-12-18.json"

TOP_K = 5  # عدد النتائج المسترجعة من RAG

DISCLAIMER = (
    "⚠️ Disclaimer: This chatbot provides general nutrition & lifestyle guidance only and does not provide medical advice. "
    "For medical conditions, medications, or symptoms, consult a qualified healthcare professional."
)

# كلمات تدل على أسئلة طبية (نرفضها)
MEDICAL_KEYWORDS = [
    "diabetes", "blood pressure", "hypertension",
    "medication", "medicine", "drug",
    "treatment", "dose", "prescription",
    "disease", "illness"
]


# ============================================================
# 2) دوال مساعدة للسلامة (رفض الأسئلة الطبية)
# ============================================================
def is_medical_question(text: str) -> bool:
    text = (text or "").lower()
    return any(k in text for k in MEDICAL_KEYWORDS)


# ============================================================
# 3) قراءة ملف FoodData Central وتحويله لنصوص معرفة (Knowledge)
#    الهدف: نحول كل عنصر غذائي إلى سطر نصي:
#    Food: X. Per 100g: Calories..., Protein..., Fat..., Carbs...
# ============================================================
def load_food_knowledge(json_path: str) -> list[str]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"JSON file not found: {json_path}\n"
            "Put the JSON file in the same folder as app.py or update JSON_PATH."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get("FoundationFoods", [])
    docs = []

    for food in foods:
        name = food.get("description", "Unknown food")
        nutrients = food.get("foodNutrients", [])

        cal = pro = fat = carb = None

        for n in nutrients:
            nut = (n.get("nutrient", {}) or {})
            nm = (nut.get("name", "")).strip().lower()
            unit = (nut.get("unitName", "")).strip().lower()
            amt = n.get("amount", None)

            # Calories (Energy in kcal)
            if "energy" in nm and unit == "kcal":
                cal = amt

            # Protein (g)
            if nm == "protein" and unit == "g":
                pro = amt

            # Fat (g)
            if "total lipid (fat)" in nm and unit == "g":
                fat = amt

            # Carbs (g) - قد يجي أكثر من اسم
            if "carbohydrate" in nm and unit == "g":
                carb = amt

        # نحتاج على الأقل سعرات حتى يكون السجل مفيد
        if cal is None:
            continue

        # نكتبها كنص معرفة واضح (لـ RAG)
        text = (
            f"Food: {name}. "
            f"Per 100g: Calories {cal} kcal, "
            f"Protein {pro} g, Fat {fat} g, Carbs {carb} g."
        )
        docs.append(text)

    return docs


# ============================================================
# 4) بناء Embeddings + FAISS Index (RAG Engine)
# ============================================================
def build_rag_index(docs: list[str]):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return embedder, index, docs


# ============================================================
# 5) Retrieval: استرجاع أفضل TOP_K مقاطع معرفة حسب سؤال المستخدم
# ============================================================
def retrieve_food_context(query: str, top_k: int = TOP_K) -> list[str]:
    q_emb = EMBEDDER.encode([query]).astype("float32")
    distances, ids = FOOD_INDEX.search(q_emb, top_k)
    return [FOOD_TEXTS[i] for i in ids[0] if i != -1]


# ============================================================
# 6) User Profile: تخزين معلومات المستخدم واستخدامها في الرد
# ============================================================
def format_profile(profile: dict) -> str:
    if not profile:
        return "No profile provided."

    fields = []
    mapping = [
        ("age", "Age"),
        ("activity", "Activity level"),
        ("goal", "Goal"),
        ("diet", "Diet preference"),
        ("allergies", "Allergies/intolerances"),
        ("meals_per_day", "Meals per day"),
        ("cooking", "Cooking time"),
        ("budget", "Budget/constraints"),
        ("notes", "Other notes"),
    ]
    for key, label in mapping:
        val = (profile.get(key) or "").strip()
        if val:
            fields.append(f"{label}: {val}")

    return "\n".join(fields) if fields else "No profile provided."


def save_profile(age, activity, goal, diet, allergies, meals_per_day, cooking, budget, notes):
    profile = {
        "age": (age or "").strip(),
        "activity": activity or "",
        "goal": goal or "",
        "diet": diet or "",
        "allergies": (allergies or "").strip(),
        "meals_per_day": meals_per_day or "",
        "cooking": cooking or "",
        "budget": (budget or "").strip(),
        "notes": (notes or "").strip(),
    }
    status = "✅ تم حفظ البيانات. الآن اكتبي في الشات: «اعطني خطة يومية» أو «اعطني خطة أسبوعية»."
    return profile, status


# ============================================================
# 7) بناء Prompt النهائي (Profile + RAG Context + User Question)
# ============================================================
def build_prompt(message: str, profile: dict, rag_context: list[str]) -> str:
    prof = format_profile(profile)
    ctx = "\n".join(rag_context) if rag_context else "No matching food data found."

    return f"""
You are an AI Virtual Health Coach.
Your role:
- Provide general nutrition and healthy lifestyle guidance.
- Create simple meal plans based on the user's preferences.

Important rules:
- No medical advice, no diagnosis, no medications.
- Avoid extreme dieting, starvation, or harmful weight-loss instructions.
- Keep it friendly, simple, and practical.
- For nutrition facts (calories/macros), use ONLY the Food Data section. Do NOT guess numbers.
- If information is missing, ask 1-2 short follow-up questions.

Food Data (Retrieved):
{ctx}

User Profile:
{prof}

User message:
{message}

Answer:
""".strip()


# ============================================================
# 8) دالة الشات (Gradio ChatInterface)
# ============================================================
def chat(message, history, profile_state, show_sources):
    message = (message or "").strip()
    profile_state = profile_state or {}

    # ✅ إذا الرسالة فاضية أو مجرد تحية
    if message == "" or message.lower() in ["hi", "hello", "hey"]:
        return (
            "Hello! I'm your virtual health coach 👋\n\n"
            "Tell me your goal or ask about a food, for example:\n"
            "- What are the calories in hummus per 100g?\n"
            "- Give me a daily meal plan\n\n"
            + DISCLAIMER
        )

    # 🛑 رفض الأسئلة الطبية
    if is_medical_question(message):
        return (
            "I’m sorry, but I can’t help with medical advice or medication recommendations. "
            "Please consult a qualified healthcare professional.\n\n" + DISCLAIMER
        )

    # إذا يطلب خطة وهو ما حفظ الهدف
    wants_plan = any(w in message.lower() for w in ["plan", "meal plan", "diet plan", "خطة", "جدول", "نظام"])
    if wants_plan and not (profile_state.get("goal") or "").strip():
        return "قبل ما أطلع لك خطة، عبّي خانة (Goal) واضغطي Save Profile. ✅\n\n" + DISCLAIMER

    # ✅ RAG Retrieval
    rag_context = retrieve_food_context(message, top_k=TOP_K)

    prompt = build_prompt(message, profile_state, rag_context)

    # ✅ Ollama Generation
    resp = ollama.generate(model=MODEL_NAME, prompt=prompt)
    answer = (resp.get("response") or "").strip()

    # ✅ عرض المصادر (اختياري)
    if show_sources and rag_context:
        sources_text = "\n".join([f"- {x}" for x in rag_context[:TOP_K]])
        answer += "\n\n📌 Sources (retrieved):\n" + sources_text

    return answer + "\n\n" + DISCLAIMER


# ============================================================
# 9) تحميل المعرفة وبناء الـ RAG مرة واحدة عند التشغيل
# ============================================================
FOOD_DOCS = load_food_knowledge(JSON_PATH)
EMBEDDER, FOOD_INDEX, FOOD_TEXTS = build_rag_index(FOOD_DOCS)
print(f"✅ Loaded foods knowledge: {len(FOOD_TEXTS)} records")


# ============================================================
# 10) واجهة Gradio (Profile + Chat)
# ============================================================
with gr.Blocks(title="Nutrition Health Coach (RAG + Ollama)") as demo:
    gr.Markdown("## Nutrition Health Coach (RAG + Ollama)\n"
                "أدخلي معلوماتك أولًا ثم اطلبي خطة. الماكروز/السعرات تُسترجع من FoodData Central.")

    profile_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1) معلومات المستخدم (Profile)")
            age = gr.Textbox(label="العمر (اختياري)")
            activity = gr.Dropdown(
                choices=["Low (mostly sitting)", "Moderate (some activity)", "High (very active)"],
                value="Moderate (some activity)",
                label="مستوى النشاط"
            )
            goal = gr.Dropdown(
                choices=[
                    "Balanced healthy eating",
                    "More energy & focus",
                    "Build healthy habits",
                    "Muscle support (general)",
                    "Sports performance (general)"
                ],
                value="Balanced healthy eating",
                label="الهدف"
            )
            diet = gr.Dropdown(
                choices=["No preference", "Balanced", "High-protein", "Vegetarian", "Vegan", "Halal-friendly"],
                value="Balanced",
                label="التفضيلات الغذائية"
            )
            allergies = gr.Textbox(label="حساسية/عدم تحمّل (مثال: lactose, nuts)")
            meals_per_day = gr.Dropdown(choices=["2", "3", "4", "5"], value="3", label="عدد الوجبات باليوم")
            cooking = gr.Dropdown(
                choices=["Quick (0-15 min)", "Medium (15-30 min)", "Long (30+ min)"],
                value="Medium (15-30 min)",
                label="وقت الطبخ"
            )
            budget = gr.Textbox(label="ميزانية/قيود (اختياري) (مثال: وجبات سريعة، طبخ قليل)")
            notes = gr.Textbox(label="ملاحظات (اختياري) (مثال: أطعمة ما أحبها)")

            show_sources = gr.Checkbox(label="Show retrieved sources (RAG)", value=True)

            save_btn = gr.Button("Save Profile")
            save_status = gr.Markdown("")

            save_btn.click(
                fn=save_profile,
                inputs=[age, activity, goal, diet, allergies, meals_per_day, cooking, budget, notes],
                outputs=[profile_state, save_status]
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2) الشات")
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[profile_state, show_sources],
                title="Chat",
                description="بعد الحفظ اكتبي: «اعطني خطة يومية» أو «اعطني خطة أسبوعية»."
            )

demo.launch()
