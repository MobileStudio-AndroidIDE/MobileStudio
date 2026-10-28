plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.godotengine.godot"
    compileSdk = 36
    buildToolsVersion = "36.1.0"

    defaultConfig {
        minSdk = 26
        manifestPlaceholders["godotLibraryVersion"] = "4.7.2.stable"
        missingDimensionStrategy("products", "editor")
    }

    flavorDimensions += "products"
    productFlavors {
        create("editor") {
            dimension = "products"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        aidl = true
        buildConfig = true
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/java", "../third_party/godot/android/lib/src/main/java")
            res.srcDirs("src/main/res", "../third_party/godot/android/lib/src/main/res")
            jniLibs.srcDirs("../third_party/godot/native")
        }
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/LICENSE",
                "META-INF/NOTICE",
                "META-INF/AL2.0",
                "META-INF/LGPL2.1"
            )
        }
    }
}

dependencies {
    implementation("androidx.fragment:fragment:1.8.6")
    implementation("androidx.documentfile:documentfile:1.1.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.annotation:annotation:1.9.1")
}
