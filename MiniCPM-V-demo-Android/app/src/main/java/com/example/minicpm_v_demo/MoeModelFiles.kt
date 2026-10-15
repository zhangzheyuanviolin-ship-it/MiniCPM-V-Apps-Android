package com.example.minicpm_v_demo

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

/** The complete, revision-pinned LES split. Files are streamed once into their final directory. */
internal object MoeModelFiles {
    private data class Asset(val path: String, val size: Long, val sha256: String, val sources: List<String>)
    @Volatile private var cached: List<Asset>? = null

    private fun catalogue(context: Context): List<Asset> = cached ?: synchronized(this) {
        cached ?: context.assets.open("minicpm-v47-files.json").bufferedReader().use { reader ->
            val root = JSONObject(reader.readText())
            val files = root.getJSONArray("files")
            List(files.length()) { index ->
                val f = files.getJSONObject(index)
                val path = f.getString("path")
                require(DownloadIntegrity.safePath(path)) { "Invalid model asset path" }
                val urls = f.getJSONArray("sources")
                Asset(path, f.getLong("size"), f.getString("sha256"),
                    List(urls.length()) { urls.getString(it) })
            }.also { result ->
                require(result.size == 43 && result.count { it.path.endsWith(".bin") } == 40)
                cached = result
            }
        }
    }

    fun ready(context: Context, directory: File): Boolean = catalogue(context).all {
        val file = File(directory, it.path)
        file.isFile && file.length() == it.size
    }

    suspend fun download(context: Context, directory: File, progress: (String) -> Unit) = withContext(Dispatchers.IO) {
        check(directory.isDirectory || directory.mkdirs()) { "Cannot create model directory" }
        val assets = catalogue(context)
        val remaining = assets.sumOf { asset ->
            val target = File(directory, asset.path)
            val part = File(directory, asset.path + ".partial")
            if (target.length() == asset.size) 0L else
                asset.size - part.length().coerceIn(0L, asset.size)
        }
        val required = remaining + 512L * 1024 * 1024
        check(StatFs(directory.absolutePath).availableBytes >= required) {
            context.getString(R.string.moe_need_space, (required + (1L shl 30) - 1) / (1L shl 30))
        }
        for ((index, asset) in assets.withIndex()) {
            coroutineContext.ensureActive()
            val target = File(directory, asset.path)
            val partial = File(directory, asset.path + ".partial")
            check(target.parentFile!!.isDirectory || target.parentFile!!.mkdirs())
            progress(context.getString(R.string.moe_check_file, index + 1, assets.size, asset.path))
            if (target.isFile && target.length() == asset.size && DownloadIntegrity.sha256(target) == asset.sha256) {
                partial.delete()
                continue
            }
            if (target.exists()) check(target.delete()) { "Cannot replace ${asset.path}" }
            if (partial.length() > asset.size) check(partial.delete())
            var succeeded = false
            var failure: Exception? = null
            for (attempt in 0 until asset.sources.size * 2) {
                coroutineContext.ensureActive()
                try {
                    if (partial.length() != asset.size) {
                        transfer(asset, asset.sources[attempt % asset.sources.size], partial) { done ->
                            progress(context.getString(R.string.moe_download_progress,
                                index + 1, assets.size, asset.path, (100 * done / asset.size).toInt()))
                        }
                    }
                    check(partial.length() == asset.size) { "Incomplete ${asset.path}" }
                    progress(context.getString(R.string.moe_check_file, index + 1, assets.size, asset.path))
                    if (DownloadIntegrity.sha256(partial) != asset.sha256) {
                        partial.delete()
                        throw IOException("SHA-256 mismatch: ${asset.path}")
                    }
                    check(partial.renameTo(target)) { "Cannot finalize ${asset.path}" }
                    succeeded = true
                    break
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    failure = e
                }
            }
            if (!succeeded) throw IOException("${asset.path}: ${failure?.message}", failure)
        }
        check(ready(context, directory)) { "Model assets incomplete" }
        progress(context.getString(R.string.download_all_complete))
    }

    private suspend fun transfer(asset: Asset, source: String, partial: File, progress: (Long) -> Unit) {
        val start = partial.length()
        val connection = (URL(source).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20000
            readTimeout = 120000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "MiniCPMV-Android/2.6")
            setRequestProperty("Accept-Encoding", "identity")
            if (start > 0) setRequestProperty("Range", "bytes=$start-")
        }
        try {
            val response = connection.responseCode
            val resume = response == HttpURLConnection.HTTP_PARTIAL
            if (response != HttpURLConnection.HTTP_OK && !resume) throw IOException("HTTP $response")
            if (resume && !DownloadIntegrity.matchesRange(connection.getHeaderField("Content-Range"), start, asset.size)) {
                throw IOException("Invalid Content-Range for ${asset.path}")
            }
            var done = if (resume) start else 0L
            var updated = 0L
            connection.inputStream.use { input ->
                FileOutputStream(partial, resume).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        if (done + count > asset.size) throw IOException("Oversized ${asset.path}")
                        output.write(buffer, 0, count)
                        done += count
                        val now = System.currentTimeMillis()
                        if (now - updated >= 750) { updated = now; progress(done) }
                    }
                }
            }
            if (done != asset.size) throw IOException("Truncated ${asset.path}: $done/${asset.size}")
        } finally {
            connection.disconnect()
        }
    }
}
