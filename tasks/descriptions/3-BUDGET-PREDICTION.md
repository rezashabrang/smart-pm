<div style="direction: rtl; text-align: right;">

### **ساخت و تنظیم مدل‌های یادگیری ماشین برای پیش‌بینی بودجه**

---

### **هدف**

هدف این وظیفه توسعه مدل‌های یادگیری ماشین برای پیش‌بینی بودجه پروژه‌ها با استفاده از مجموعه داده‌های DSLIB است. شما باید از روش‌های تنظیم ابرپارامتر (به جز تنظیم دستی) برای بهینه‌سازی عملکرد مدل استفاده کنید.

---

### **دستورالعمل‌ها**

#### **مرحله 1: انتخاب و آماده‌سازی داده‌ها**

1. **انتخاب فایل‌ها:**

   - یک یا چند فایل از کتابخانه DSLIB انتخاب کنید که حاوی داده‌های مرتبط با مدیریت پروژه باشند.
   - انتخاب فایل‌ها و برگه‌ها را بر اساس ویژگی‌های موجود (مانند اندازه پروژه، اندازه تیم، هزینه، مدت زمان) توجیه کنید.

2. **آماده‌سازی داده‌ها:**
   - استخراج ویژگی‌های معنادار.
   - مدیریت مقادیر گمشده، مقادیر پرت، و مقیاس‌بندی ویژگی‌های عددی با استفاده از استانداردسازی یا مقیاس‌بندی Min-Max.
   - رمزگذاری متغیرهای دسته‌ای با استفاده از One-Hot Encoding یا Label Encoding.

---

#### **مرحله 2: توسعه مدل**

1. **انتخاب مدل‌ها:**

   - حداقل دو مدل یادگیری ماشین از فهرست زیر ایجاد کنید:
     - رگرسیون خطی
     - درخت‌های تصمیم
     - جنگل تصادفی
     - گرادیان بوستینگ (مانند XGBoost، LightGBM)
     - شبکه‌های عصبی

2. **آموزش مدل‌ها:**
   - مدل‌ها را با مجموعه داده آماده‌شده آموزش داده و عملکرد آن‌ها را با استفاده از معیارهایی مانند خطای میانگین مربعات (MSE) یا نمره R² ارزیابی کنید.

---

#### **مرحله 3: تنظیم ابرپارامتر**

1. **انتخاب یک روش تنظیم:**

   - از یک یا چند مورد از روش‌های زیر برای تنظیم ابرپارامترها استفاده کنید:
     - **جستجوی شبکه‌ای (Grid Search)**
     - **جستجوی تصادفی (Random Search)**
     - **بهینه‌سازی بیزی (Bayesian Optimization)**
     - **Hyperband**

2. **بهینه‌سازی ابرپارامترها:**
   - ابرپارامترهایی مانند نرخ یادگیری، عمق درخت، تعداد تخمین‌گرها را انتخاب کنید.
   - تنظیم را انجام داده و بهترین ترکیب ابرپارامترها را گزارش کنید.

---

#### **مرحله 4: گزارش‌دهی**

1. **مقایسه مدل‌ها:**

   - عملکرد مدل‌ها را قبل و بعد از تنظیم مقایسه کنید.
   - نشان دهید که چگونه تنظیم، نتایج را بهبود داده است.

2. **بصری‌سازی:**

   - حداقل سه نمودار شامل موارد زیر اضافه کنید:
     - نمودار سطح سه‌بعدی از نتایج جستجوی شبکه‌ای یا تصادفی.
     - نمودار اهمیت ویژگی‌ها (مانند نمودار میله‌ای برای جنگل تصادفی یا XGBoost).
     - مقایسه عملکرد مدل‌ها قبل و بعد از تنظیم (مانند نمودار خطی MSE).

3. **بینش‌ها:**
   - نکات کلیدی و درس‌های آموخته‌شده در طول وظیفه را خلاصه کنید.

---

### **تحویل‌ها**

1. **کد و نتایج:**

   - اسکریپت‌ها یا نوت‌بوک‌های پایتون برای آماده‌سازی داده‌ها، ساخت مدل و تنظیم ابرپارامتر.
   - مدل‌های نهایی آموزش‌داده‌شده با پیکربندی ابرپارامترهای آن‌ها.

2. **گزارش:**

   - یک گزارش مختصر (700 تا 1000 کلمه) که شامل:
     - انتخاب و آماده‌سازی داده‌ها.
     - فرآیند توسعه مدل.
     - مراحل تنظیم ابرپارامتر و نتایج.
     - مقایسه عملکرد مدل و نتیجه‌گیری.

3. **بصری‌سازی:**
   - نمودارها را به صورت فایل‌های PNG یا جاسازی‌شده در گزارش ارائه کنید.

---

### **معیارهای ارزیابی**

1. **انتخاب و آماده‌سازی داده‌ها:**
   - کیفیت استخراج ویژگی و پیش‌پردازش داده‌ها.
2. **پیاده‌سازی مدل:**
   - صحت و تنوع مدل‌های پیاده‌سازی‌شده.
3. **تنظیم ابرپارامتر:**
   - مناسب بودن روش‌های انتخاب‌شده و بهبود نتایج.
4. **گزارش‌دهی و بصری‌سازی:**
   - وضوح، عمق و کیفیت گزارش و نمودارها.

---

</div>
