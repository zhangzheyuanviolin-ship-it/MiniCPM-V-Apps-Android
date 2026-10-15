package com.example.minicpm_v_demo

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/** A user-visible service for the lifetime of the loaded local model, including idle turns. */
class InferenceForegroundService : Service() {
    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= 26) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL, getString(R.string.inference_channel), NotificationManager.IMPORTANCE_LOW)
            )
        }
        val open = PendingIntent.getActivity(this, 70,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val unload = PendingIntent.getService(this, 71,
            Intent(this, InferenceForegroundService::class.java).setAction(ACTION_UNLOAD),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_lightbulb)
            .setContentTitle(getString(R.string.inference_notification_title))
            .setContentText(getString(R.string.inference_notification_text))
            .setContentIntent(open)
            .addAction(R.drawable.ic_delete, getString(R.string.inference_unload), unload)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setOngoing(true).setOnlyAlertOnce(true).setSilent(true).setShowWhen(false)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(4707, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else if (Build.VERSION.SDK_INT >= 29) {
            startForeground(4707, notification, 0)
        } else {
            startForeground(4707, notification)
        }
        Log.i("InferenceForeground", "Local model foreground service started")
    }

    override fun onBind(intent: Intent?): IBinder = Binder()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_UNLOAD) LlamaEngine.getInstance(applicationContext).requestUnload()
        // Never automatically reload a multi-gigabyte model after process death.
        return START_NOT_STICKY
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        LlamaEngine.getInstance(applicationContext).requestUnload()
        super.onTaskRemoved(rootIntent)
    }

    companion object {
        private const val CHANNEL = "local-model-inference"
        private const val ACTION_UNLOAD = "com.example.minicpm_v_demo.UNLOAD_LOCAL_MODEL"
    }
}

/** Explicit start plus important binding, acknowledged before native allocation begins. */
internal object InferenceServiceLease {
    private var connection: ServiceConnection? = null
    private var ready: CompletableDeferred<Unit>? = null

    suspend fun acquire(context: Context) = withContext(Dispatchers.Main.immediate) {
        ready?.let { it.await(); return@withContext }
        val app = context.applicationContext
        val intent = Intent(app, InferenceForegroundService::class.java)
        val signal = CompletableDeferred<Unit>()
        ready = signal
        val listener = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, service: IBinder) { signal.complete(Unit) }
            override fun onServiceDisconnected(name: ComponentName) {
                signal.completeExceptionally(IllegalStateException("Inference service disconnected"))
            }
            override fun onNullBinding(name: ComponentName) {
                signal.completeExceptionally(IllegalStateException("Inference service did not bind"))
            }
            override fun onBindingDied(name: ComponentName) {
                signal.completeExceptionally(IllegalStateException("Inference service binding died"))
            }
        }
        try {
            ContextCompat.startForegroundService(app, intent)
            check(app.bindService(intent, listener, Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)) {
                "Cannot bind local inference service"
            }
            connection = listener
            withTimeout(10_000) { signal.await() }
        } catch (e: Exception) {
            releaseOnMain(app)
            throw IllegalStateException(context.getString(R.string.inference_service_failed), e)
        }
    }

    suspend fun release(context: Context) = withContext(Dispatchers.Main.immediate) {
        releaseOnMain(context.applicationContext)
    }

    private fun releaseOnMain(context: Context) {
        connection?.let {
            try { context.unbindService(it) } catch (e: IllegalArgumentException) {
                Log.w("InferenceForeground", "Binding already released", e)
            }
        }
        connection = null
        ready = null
        context.stopService(Intent(context, InferenceForegroundService::class.java))
    }
}
