// Lightweight Redis client shim for development environments
// Exports initRedis, closeRedis, getRedisClient, isRedisEnabled

const isEnabled = () => {
  return (process.env.REDIS_ENABLED || 'false').toLowerCase() === 'true';
};

let redisClient = null;
let _lastRedisErrorMsg = null;

async function initRedis() {
  if (!isEnabled()) return;
  try {
    // Lazy require to avoid adding redis as a hard dependency for dev
    const Redis = require('redis');
    // Disable aggressive reconnects in dev; treat Redis as optional
    redisClient = Redis.createClient({
      url: process.env.REDIS_URL,
      socket: {
        // Return false to disable automatic reconnection attempts
        reconnectStrategy: () => false,
      },
    });

    // Non-fatal error handler: format AggregateError nicely and dedupe repeated messages
    redisClient.on('error', (err) => {
      try {
        let msg = '';
        if (typeof AggregateError !== 'undefined' && err instanceof AggregateError) {
          msg = err.errors.map(e => (e && e.message) ? e.message : String(e)).join(' | ');
        } else if (err && Array.isArray(err.errors)) {
          msg = err.errors.map(e => (e && e.message) ? e.message : String(e)).join(' | ');
        } else {
          msg = err && err.message ? err.message : String(err);
        }

        if (msg === _lastRedisErrorMsg) return; // avoid log spam
        _lastRedisErrorMsg = msg;
        console.warn('⚠️ Redis error (non-fatal):', msg);
      } catch (e) {
        if (String(e) !== _lastRedisErrorMsg) {
          _lastRedisErrorMsg = String(e);
          console.warn('⚠️ Redis error (non-fatal)');
        }
      }
    });

    await redisClient.connect();
    console.log('✅ Redis connected');
    return true;
  } catch (e) {
    // Do not throw — treat Redis as optional in development
    console.warn('⚠️ Redis init skipped or failed:', e && e.message ? e.message : String(e));
    redisClient = null;
    return false;
  }
}

async function closeRedis() {
  if (!redisClient) return;
  try {
    await redisClient.quit();
    redisClient = null;
  } catch (e) {
    console.warn('⚠️ Redis close failed:', e.message);
  }
}

function getRedisClient() {
  // Return the client if available, otherwise null. Callers should handle null.
  if (!isEnabled()) return null;
  return redisClient;
}

function isRedisEnabled() {
  return isEnabled();
}

module.exports = {
  initRedis,
  closeRedis,
  getRedisClient,
  isRedisEnabled
};
