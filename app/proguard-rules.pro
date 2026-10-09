# proguard-rules.pro
#
# MobileStudio ProGuard 규칙
#
# JNI 메서드는 ProGuard 로 난독화하면 안 됨
# → NativeBridge 의 모든 native 메서드 보존

# NativeBridge JNI 클래스 완전 보존
-keep class com.example.mobilestudio.NativeBridge {
    *;
}

# native 메서드가 있는 클래스 전부 보존
-keepclasseswithmembernames class * {
    native <methods>;
}

# Kotlin object (싱글턴) 보존
-keep class com.example.mobilestudio.StorageHelper { *; }

# LineLevel enum 보존
-keepclassmembers enum com.example.mobilestudio.LineLevel {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# XZ 라이브러리 보존
-keep class org.tukaani.xz.** { *; }

# Kotlin 코루틴
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}