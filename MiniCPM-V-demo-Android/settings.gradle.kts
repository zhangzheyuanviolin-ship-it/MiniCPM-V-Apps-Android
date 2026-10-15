pluginManagement {
    repositories {
        // Use official repositories for reproducible cloud builds.
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        // Official Android and Maven repositories.
        google()
        mavenCentral()
    }
}

rootProject.name = "MiniCPM-V-demo"
include(":app")
