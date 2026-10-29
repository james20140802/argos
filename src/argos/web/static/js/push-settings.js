/* Morning-briefing notification toggle on the settings page (ARG-321).
 *
 * The truth is split in two: the browser holds this device's push
 * subscription, the server holds the list it sends to. On load we read the
 * browser side and make sure the server agrees (re-registering if the row was
 * dropped), then render 켜짐/꺼짐. The toggle subscribes with the server's
 * VAPID key or unsubscribes, keeping both sides in step.
 *
 * iOS only offers Web Push to a web app added to the home screen, and every
 * browser wants a secure context — both cases render an explanation instead
 * of a dead button. Endpoints are only ever sent in POST bodies.
 */
(function () {
  'use strict';

  var root = document.getElementById('push-settings');
  if (!root) return;
  var stateEl = document.getElementById('push-state');
  var hintEl = document.getElementById('push-hint');
  var button = document.getElementById('push-toggle');

  var publicKey = null;
  var subscribed = false;
  var busy = false;

  function render(state, hint) {
    var labels = {
      on: '켜짐',
      off: '꺼짐',
      denied: '알림이 차단됨',
      unsupported: '사용할 수 없음',
      error: '확인 실패',
      loading: '확인 중…',
    };
    stateEl.textContent = labels[state] || state;
    stateEl.setAttribute('data-push-state', state);
    if (hint) {
      hintEl.textContent = hint;
      hintEl.hidden = false;
    } else {
      hintEl.textContent = '';
      hintEl.hidden = true;
    }
    var actionable = state === 'on' || state === 'off' || state === 'error';
    button.hidden = !actionable;
    button.disabled = busy;
    button.textContent = state === 'on' ? '알림 끄기' : '아침 브리핑 알림 받기';
  }

  function isIos() {
    return /iPhone|iPad|iPod/.test(navigator.userAgent);
  }

  function isStandalone() {
    return (
      window.navigator.standalone === true ||
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches)
    );
  }

  function keyBytes(b64url) {
    var pad = '='.repeat((4 - (b64url.length % 4)) % 4);
    var raw = atob((b64url + pad).replace(/-/g, '+').replace(/_/g, '/'));
    var out = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  }

  function post(path, body) {
    return fetch(path, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(function (res) {
      if (!res.ok) throw new Error(path + ' → HTTP ' + res.status);
      return res.json();
    });
  }

  function registration() {
    return navigator.serviceWorker.getRegistration('/').then(function (reg) {
      return reg || navigator.serviceWorker.register('/sw.js', { scope: '/' });
    }).then(function () {
      return navigator.serviceWorker.ready;
    });
  }

  function loadKey() {
    return fetch('/push/key', { credentials: 'same-origin' })
      .then(function (res) {
        if (!res.ok) throw new Error('/push/key → HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        publicKey = data.publicKey;
      });
  }

  function register(sub) {
    var json = sub.toJSON();
    return post('/push/subscribe', { endpoint: json.endpoint, keys: json.keys });
  }

  function refresh() {
    if (Notification.permission === 'denied') {
      render('denied', '기기(브라우저) 설정에서 이 앱의 알림을 허용하면 켤 수 있습니다.');
      return Promise.resolve();
    }
    return registration()
      .then(function (reg) {
        return reg.pushManager.getSubscription();
      })
      .then(function (sub) {
        if (!sub || Notification.permission !== 'granted') {
          subscribed = false;
          return;
        }
        return post('/push/status', { endpoint: sub.endpoint }).then(function (data) {
          subscribed = true;
          // Server forgot this device (expired-cleanup, DB reset): re-register.
          if (!data.subscribed) return register(sub);
        });
      })
      .then(function () {
        render(subscribed ? 'on' : 'off');
      });
  }

  function turnOn() {
    // requestPermission first, straight from the click: Safari only shows
    // the prompt while the user gesture is still active.
    return Notification.requestPermission()
      .then(function (permission) {
        if (permission !== 'granted') {
          subscribed = false;
          return refresh();
        }
        return (publicKey ? Promise.resolve() : loadKey())
          .then(registration)
          .then(function (reg) {
            return reg.pushManager.getSubscription().then(function (existing) {
              return (
                existing ||
                reg.pushManager.subscribe({
                  userVisibleOnly: true,
                  applicationServerKey: keyBytes(publicKey),
                })
              );
            });
          })
          .then(register)
          .then(function () {
            subscribed = true;
            render('on');
          });
      });
  }

  function turnOff() {
    return registration()
      .then(function (reg) {
        return reg.pushManager.getSubscription();
      })
      .then(function (sub) {
        if (!sub) return;
        var endpoint = sub.endpoint;
        return sub.unsubscribe().then(function () {
          return post('/push/unsubscribe', { endpoint: endpoint });
        });
      })
      .then(function () {
        subscribed = false;
        render('off');
      });
  }

  function fail(err) {
    if (window.console && console.warn) console.warn('[argos] push settings:', err);
    render('error', '알림 설정을 바꾸지 못했습니다. 잠시 뒤 다시 시도해 주세요.');
  }

  if (
    !window.isSecureContext ||
    !('serviceWorker' in navigator) ||
    !('PushManager' in window) ||
    !('Notification' in window)
  ) {
    var why = isIos() && !isStandalone()
      ? 'Safari 공유 메뉴의 "홈 화면에 추가"로 설치한 Argos 앱에서 켤 수 있습니다.'
      : '이 브라우저(또는 HTTPS가 아닌 주소)에서는 웹 푸시를 쓸 수 없습니다.';
    render('unsupported', why);
    return;
  }

  button.addEventListener('click', function () {
    if (busy) return;
    busy = true;
    button.disabled = true;
    var action = subscribed ? turnOff() : turnOn();
    action.catch(fail).then(function () {
      busy = false;
      button.disabled = false;
    });
  });

  loadKey().catch(function () {});
  refresh().catch(fail);
})();
