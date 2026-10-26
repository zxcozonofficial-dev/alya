"""Alya — CLI.

  python main.py synth      --out data/synth --count 200
  python main.py bc-data    --songs "C:/osu!/Songs" --replays "C:/osu!/Replays" --out data/bc.npz
  python main.py bc         --data data/bc.npz --out ckpt/bc.pt
  python main.py ppo        --maps "C:/osu!/Songs" --init ckpt/bc.pt --out ckpt/ppo.pt
  python main.py export     --model ckpt/ppo.pt --map song.osu --out alya.osr
  python main.py live       --model ckpt/alya.pt
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Alya: нейросеть для osu!standard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="сгенерировать случайные карты")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--count", type=int, default=200)

    p = sub.add_parser("bc-data", help="реплеи .osr -> датасет для BC")
    p.add_argument("--songs", nargs="*", default=[], help="папка Songs (для поиска карт по md5)")
    p.add_argument("--replays", nargs="*", default=[], help=".osr файлы или папки")
    p.add_argument("--autoplay-maps", nargs="*", default=[], help="добавить автоплей по этим картам")
    p.add_argument("--out", default="data/bc.npz")
    p.add_argument("--limit", type=int, help="макс. реплеев")

    p = sub.add_parser("bc", help="behavior cloning")
    p.add_argument("--data", default="data/bc.npz")
    p.add_argument("--out", default="ckpt/bc.pt")
    p.add_argument("--init")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=256)

    p = sub.add_parser("ppo", help="дообучение PPO в симуляторе")
    p.add_argument("--maps", nargs="+", required=True, help=".osu файлы / папки (Songs)")
    p.add_argument("--limit", type=int, default=500, help="макс. карт в пуле")
    p.add_argument("--init")
    p.add_argument("--out", default="ckpt/ppo.pt")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mods", nargs="*", default=["NM"],
                   help="моды для аугментации, на каждую карту случайный: NM HR DT HDDT ...")
    p.add_argument("--kl", type=float, default=0.05, help="якорь к BC-стилю (0 = чистый PPO, робот)")
    p.add_argument("--ref", help="модель-якорь стиля (по умолчанию --init); при дообучении PPO укажите BC")
    p.add_argument("--wasted", type=float, default=0.15, help="штраф за клик, не задевший ноту")
    p.add_argument("--smooth", type=float, default=0.0, help="штраф за резкие углы аима (разворот на скорости)")
    p.add_argument("--hard", type=float, default=0.0, help="доля эпизодов, начинающихся на стримах и двойках 1/8")
    p.add_argument("--cp-cost", type=float, default=0.1, help="штраф за сорванный тик/повтор/конец слайдера")
    p.add_argument("--timing", type=float, default=0.0, help="штраф за ошибку по времени (в долях окна 300) за попадание")

    p = sub.add_parser("export", help="сыграть карту в симуляторе и сохранить .osr")
    p.add_argument("--model", help="чекпойнт; без него — автоплей")
    p.add_argument("--map", required=True)
    p.add_argument("--out", default="alya.osr")
    p.add_argument("--player", default="Alya")
    p.add_argument("--sample", action="store_true", help="стохастическая политика")
    p.add_argument("--mods", default="", help="например HDDT, HR, EZHT")
    p.add_argument("--acc", type=float, default=0.97, help="уровень: точность игрока, которого изображать")
    p.add_argument("--miss", type=float, default=0.005, help="уровень: доля промахов")

    p = sub.add_parser("live", help="играть в osu! через tosu + SendInput (только оффлайн)")
    p.add_argument("--model", required=True)
    p.add_argument("--curve", type=float, default=0.2, help="изгиб траектории между нотами (0 — по прямой)")
    p.add_argument("--smoke-key", default="c", help="клавиша дыма для рисования в перерывах; \"\" — не рисовать")
    p.add_argument("--keys", default="zx")
    p.add_argument("--port", type=int, default=24050)
    p.add_argument("--acc", type=float, default=0.97)
    p.add_argument("--miss", type=float, default=0.005)
    p.add_argument("--dry-run", action="store_true", help="не трогать мышь и клавиатуру, только проверить связь")
    p.add_argument("--offset-y", type=float, default=0.0, help="поправка положения поля по вертикали, osu!px")
    p.add_argument("--no-auto-offset", action="store_true", help="не подстраивать тайминг по ошибкам попаданий")
    p.add_argument("--offset-ms", type=float, default=0.0,
                   help="поправка времени: модель жмёт поздно — ставь +5..+20, рано — минус")

    p = sub.add_parser("chat", help="поговорить с Алей в терминале")
    p.add_argument("--backend", default="gemini-lite,gemini,gemini2-lite,gemini2,ollama",
                   help="claude | gemini | gemini-lite | gemini2 | gemini2-lite | groq | ollama "
                        "или цепочка запасных через запятую")
    p.add_argument("--model", help="модель первого бэкенда (по умолчанию своя у каждого)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"], help="только для claude")
    p.add_argument("--new", action="store_true", help="начать без прошлых реплик (память фактов остаётся)")
    p.add_argument("--lang", default="en", choices=["en", "ru"], help="на каком языке она отвечает")
    p.add_argument("--voice", default="en-US-AnaNeural", help="голос озвучки edge-tts (off — без голоса)")
    p.add_argument("--pitch", default="+0Hz", help="высота голоса, напр. +25Hz")
    p.add_argument("--rate", default="+0%", help="скорость речи, напр. +8%%")
    p.add_argument("--vts", action="store_true", help="вести модельку в VTube Studio (рот, эмоции)")
    p.add_argument("--game", default="alya", choices=["alya", "creator", "off"],
                   help="комментировать игру из tosu: alya — играет бот, creator — играешь ты, off — нет")
    p.add_argument("--twitch", help="канал Twitch, чат которого она читает (без токена, только чтение)")
    p.add_argument("--overlay", type=int, default=24080, help="порт оверлея для OBS (0 — выключить)")

    sub.add_parser("obs-setup", help="собрать сцены Али в OBS (нужен включённый сервер WebSocket)")
    sub.add_parser("stream", help="запустить всё для стрима: tosu, бота и Алю (настройки в stream.json)")

    p = sub.add_parser("calibrate", help="обвести курсором углы игрового поля для проверки")
    p.add_argument("--port", type=int, default=24050)
    p.add_argument("--offset-y", type=float, default=0.0)

    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd in ("bc-data", "bc", "ppo"):
        # обучение — с пониженным приоритетом, чтобы не сбивать тайминги живой игры рядом
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

    if a.cmd == "synth":
        from alya.synth import write_maps
        print(f"{len(write_maps(a.out, a.count))} карт -> {a.out}")

    elif a.cmd == "bc-data":
        from alya.bc import build_dataset
        from alya.data import load_maps, md5_index
        idx = {}
        for s in a.songs:
            idx.update(md5_index(s, cache=str(Path("data") / f"md5_{hashlib.md5(s.encode()).hexdigest()[:8]}.json")))
        reps = []
        for r in a.replays:
            r = Path(r)
            reps += sorted(map(str, r.rglob("*.osr"))) if r.is_dir() else [str(r)]
        reps = reps[:a.limit] if a.limit else reps
        auto = load_maps(a.autoplay_maps) if a.autoplay_maps else []
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        build_dataset(reps, idx, a.out, auto)

    elif a.cmd == "bc":
        from alya.bc import train_bc
        train_bc(a.data, a.out, a.init, a.steps, a.batch, a.seq, device=dev)

    elif a.cmd == "ppo":
        from alya.data import load_maps
        from alya.ppo import train_ppo
        from alya.beatmap import mods_from_string
        maps = load_maps(a.maps, limit=a.limit,
                         mods_choices=[mods_from_string(m) for m in a.mods])
        train_ppo(maps, a.out, a.init, a.envs, a.horizon, a.updates, a.lr, device=dev, kl_coef=a.kl,
                  ref_path=a.ref, wasted_cost=a.wasted, smooth_cost=a.smooth,
                  hard_frac=a.hard, cp_cost=a.cp_cost, timing_cost=a.timing)

    elif a.cmd == "export":
        from alya import model as M
        from alya.beatmap import mods_from_string, parse_osu
        from alya.export import export
        bm = parse_osu(a.map, mods_from_string(a.mods))
        net = M.load(a.model, dev).eval() if a.model else None
        r, acc = export(bm, net, a.out, not a.sample, dev, a.player, a.acc, a.miss)
        print(f"{bm}\n300 {r.n300} | 100 {r.n100} | 50 {r.n50} | miss {r.nmiss} | "
              f"combo {r.max_combo} | acc {acc*100:.2f}% -> {a.out}")

    elif a.cmd == "live":
        from alya.live import run_live
        run_live(a.model, tuple(a.keys), dev, a.port, acc=a.acc, miss=a.miss,
                 dry_run=a.dry_run, offset_y=a.offset_y, offset_ms=a.offset_ms,
                 auto_offset=not a.no_auto_offset, smoke_key=a.smoke_key or None,
                 curve=a.curve)

    elif a.cmd == "chat":
        from alya.brain import chat
        chat(a.backend, a.model, a.effort, a.new, a.lang, None if a.voice == "off" else a.voice,
             a.vts, a.pitch, a.rate, None if a.game == "off" else a.game, a.twitch,
             None if a.overlay == 0 else a.overlay)

    elif a.cmd == "obs-setup":
        from alya.obs_setup import setup
        setup()

    elif a.cmd == "stream":
        from alya.stream import run
        run()

    elif a.cmd == "calibrate":
        from alya.live import calibrate
        calibrate(a.port, a.offset_y)


if __name__ == "__main__":
    sys.exit(main())
