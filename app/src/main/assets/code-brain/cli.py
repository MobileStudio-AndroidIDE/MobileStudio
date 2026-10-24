"""CLI interface for RENA-BR."""
from __future__ import annotations
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.code_brain import CodeBrain
from code.ir.ir_types import IRNode, IRNodeType


# ── The conversation SESSION ─────────────────────────────────────────
# A CLI PROCESS is one turn of a conversation, never the conversation
# itself.  With `--session NAME` (or the `session` subcommand) the same
# discourse — the topic state and the record the references resolve
# against — is loaded before the turn and saved after it, so the next
# process continues where this one left off.  Without it, every
# invocation is a fresh conversation.
_SESSIONS_DIR = os.path.join(os.path.expanduser("~"), ".rena_brain",
                             "sessions")


def _session_path(name: str) -> str:
    """The session file for a name — names are file-safe, never paths."""
    safe = "".join(c for c in (name or "")
                   if c.isalnum() or c in "._-").strip("._")
    return os.path.join(_SESSIONS_DIR, (safe or "session") + ".json")


def _load_session(name: str, brain) -> dict:
    """Resume this conversation: the SAME topic state and turn record a
    reference resolves against.  A missing or unreadable record resumes
    as an empty conversation — never an error and never a guess."""
    try:
        with open(_session_path(name), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    try:
        brain.conversation_manager.load_dict(data)
    except Exception:
        return {}
    return data


def _save_session(name: str, brain) -> str:
    """Persist the conversation for the next process.  Best effort: a
    session that cannot be written still answers this turn."""
    try:
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        with open(_session_path(name), "w", encoding="utf-8") as fh:
            json.dump(brain.conversation_manager.to_dict(), fh,
                      ensure_ascii=False, indent=2)
        return _session_path(name)
    except Exception:
        return ""


def run_session(brain, name: str, turn: str = "", fresh: bool = False) -> None:
    """Continue a conversation session: interactively, or as a single
    turn in that conversation (`--turn`), or from scratch (`--new`)."""
    if fresh:
        brain.conversation_manager.clear()
    else:
        _load_session(name, brain)
    history = brain.conversation_manager.history
    print(f"SESSION: {name} — {len(history)} turn(s) in this "
          f"conversation")
    if brain.conversation_manager.current_topic:
        print(f"TOPIC: {brain.conversation_manager.current_topic}")
    print()
    if turn:
        _session_turn(brain, name, turn)
        return
    if not sys.stdin.isatty():
        # a non-interactive invocation without a turn: state only, so
        # a session is never silently consumed by a closing pipe
        return
    print("Type a message. 'exit' or Ctrl-D ends the session.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        _session_turn(brain, name, line)


def _session_turn(brain, name: str, text: str) -> None:
    result = brain.process_natural(text)
    print(f"rena> {result.get('response') or result.get('text') or ''}"
          f"\n")
    _save_session(name, brain)


def run_rena_command(args, config, command: str) -> int:
    from cognition.rena import RenaBrain
    from cognition.autonomous import AutonomousAgent
    from cognition.actuators import ActuatorRegistry

    brain = RenaBrain(config)
    if command == "brain":
        status = brain.status()
        if getattr(args, "json", False):
            print(json.dumps(status, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"DATA: {status['data_dir']}")
            print(f"VSA: {status['dimension']} dimensions")
            print(f"CONCEPTS: {status['concepts']['concepts']} "
                  f"({status['concepts']['observations']} observations)")
            print(f"EPISODES: {status['episodes']['episodes']}")
            print(f"STAGE: {status['learning']['stage']}")
            print(f"ACTIVE: {status['active']}")
            print(f"CURIOSITY: {status['curiosity']}")
            print(f"META: {status['metacognition']}")
            for row in status["actuators"]:
                state = "available" if row["available"] else                     ("unavailable: " + row["reason"])
                print(f"  {row['name']}: {state}")
        brain.close()
        return 0
    if command == "think":
        result = brain.think(
            getattr(args, "query", ""),
            modality=getattr(args, "modality", "text"))
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"CONCEPT: {result['concept']} "
                  f"({result['concept_score']})")
            print(f"CONFIDENCE: {result['confidence']}")
            print(f"RESPONSE: {result['response']}")
            if result.get("prediction"):
                print(f"PREDICTION: {result['prediction'].get('symbol')}")
        brain.close()
        return 0
    if command in ("observe", "perceive"):
        modality = getattr(args, "modality", "text")
        data = getattr(args, "text", "") or getattr(args, "path", "")
        if not data:
            print("observe requires --text or --path", file=sys.stderr)
            brain.close()
            return 2
        result = brain.observe(modality, data,
                               label=getattr(args, "label", ""))
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"OBSERVED: {result['label']} [{result['modality']}]")
            print(f"SURPRISE: {result['active']['surprise']}")
            print(f"FREE ENERGY: {result['active']['free_energy']}")
        brain.close()
        return 0
    if command == "act":
        name = getattr(args, "actuator", "")
        operation = getattr(args, "operation", "") or ""
        request = None
        if getattr(args, "code", ""):
            request = {"operation": operation or "run",
                       "code": args.code}
        elif getattr(args, "expression", ""):
            request = {"operation": operation or "evaluate",
                       "expression": args.expression}
        elif getattr(args, "path", ""):
            request = {"operation": operation or "describe",
                       "path": args.path}
        elif getattr(args, "prompt", ""):
            request = {"operation": operation or "generate",
                       "prompt": args.prompt}
        elif getattr(args, "request", ""):
            raw_request = args.request
            if raw_request.lstrip().startswith(("{", "[")):
                try:
                    request = json.loads(raw_request)
                except json.JSONDecodeError:
                    request = raw_request
            else:
                request = raw_request
        if request is None:
            print("act requires --code/--expression/--path/--prompt",
                  file=sys.stderr)
            brain.close()
            return 2
        result = brain.act(name, request, force=getattr(args, "force", False))
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"ACT: {result.get('actuator')}."
                  f"{result.get('operation')} ok={result.get('ok')}")
            if result.get("output") is not None:
                print(f"OUTPUT: {result['output']}")
            if result.get("artifacts"):
                print("ARTIFACTS: " + ", ".join(result["artifacts"]))
            if result.get("error"):
                print(f"ERROR: {result['error']}")
        brain.close()
        return 0
    if command == "autonomous":
        agent = AutonomousAgent(config, brain=brain)
        report = agent.run(
            cycles=getattr(args, "cycles", 4),
            time_budget=getattr(args, "time_budget", None))
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"AUTONOMOUS: {report['cycles_completed']}/"
                  f"{report['cycles_requested']} cycles")
            print(f"GROWTH: {report['growth_successes']}/"
                  f"{report['growth_tasks']} verified")
            print(f"PROBES: {report['probe_successes']}/{report['probes']}")
            print(f"AUGMENT: {report.get('augmentation_adopted', 0)}/"
                  f"{report.get('augmentation_candidates', 0)} adopted")
            print(f"INVENT: {report.get('primitives_invented', 0)} "
                  f"primitives from {report.get('invention_candidates', 0)} "
                  f"candidates")
            print(f"CONSOLIDATIONS: {report['consolidations']}")
            print(f"STOPPED: {report['stopped_reason']}")
        brain.close()
        return 0
    if command == "augment":
        if getattr(args, "trials", 0):
            config.augmentation_trials = max(1, int(args.trials))
        report = brain.augment(
            cycles=getattr(args, "cycles", 1),
            candidates=getattr(args, "candidates", None),
            time_budget=getattr(args, "time_budget", None),
            domain=getattr(args, "domain", "math"))
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2,
                             default=str))
        else:
            print(f"AUGMENT: {report['cycles']}/"
                  f"{report['requested_cycles']} cycles")
            print(f"CANDIDATES: {report['candidates_generated']} "
                  f"generated, {report['candidates_verified']} verified")
            print(f"ADOPTED: {report['candidates_adopted']} "
                  f"(macros {report['macros_before']} -> "
                  f"{report['macros_after']})")
            print(f"REJECTED: {report['candidates_rejected']} "
                  f"ROLLED BACK: {report['candidates_rolled_back']}")
            print(f"STOPPED: {report['stopped_reason']}")
        brain.close()
        return 0
    if command == "invent":
        kind = getattr(args, "kind", "primitive")
        if kind == "sensor":
            report = brain.invent_sensor(
                modality=getattr(args, "modality", "text"),
                samples=list(getattr(args, "sample", []) or []),
                cycles=getattr(args, "cycles", 1),
                candidates=(getattr(args, "candidates", 0) or 12),
                time_budget=getattr(args, "time_budget", None))
        else:
            report = brain.invent_primitives(
                cycles=getattr(args, "cycles", 1),
                candidates=(getattr(args, "candidates", 0) or 6),
                time_budget=getattr(args, "time_budget", None))
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2,
                             default=str))
        elif kind == "sensor":
            print(f"SENSOR: {report['sensor'] or 'none'} "
                  f"({report['candidates_adopted']}/"
                  f"{report['candidates_generated']} adopted)")
            print(f"STOPPED: {report['stopped_reason']}")
        else:
            print(f"PRIMITIVES: {report['candidates_adopted']}/"
                  f"{report['candidates_generated']} adopted "
                  f"({report['primitives_before']} -> "
                  f"{report['primitives_after']})")
            print(f"STOPPED: {report['stopped_reason']}")
        brain.close()
        return 0
    if command == "sensor":
        action = getattr(args, "sensor_action", "list")
        if action == "invent":
            report = brain.invent_sensor(
                modality=getattr(args, "modality", "text"),
                samples=list(getattr(args, "sample", []) or []),
                cycles=getattr(args, "cycles", 1),
                candidates=(getattr(args, "candidates", 0) or 12),
                time_budget=getattr(args, "time_budget", None))
        elif action == "read":
            report = brain.read_sensor(
                getattr(args, "name", ""),
                getattr(args, "source", ""))
        else:
            report = brain.sensors.stats()
            report["sensors_detail"] = [
                spec.to_dict() for spec in brain.sensors.all()]
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2,
                             default=str))
        elif action == "read":
            print(f"SENSOR: {report['sensor']}")
            print(f"FEATURES: {report['features']}")
        elif action == "invent":
            print(f"SENSOR: {report['sensor'] or 'none'} "
                  f"({report['candidates_adopted']}/"
                  f"{report['candidates_generated']} adopted)")
        else:
            print(f"SENSORS: {report['sensors']} "
                  f"({', '.join(report['modalities']) or 'none'})")
            for row in report["sensors_detail"]:
                print(f"  {row['name']} [{row['modality']}] "
                      f"{', '.join(row['channels'])}")
        brain.close()
        return 0
    if command == "curriculum":
        rows = brain.propose_curriculum(
            count=getattr(args, "count", 6),
            domain=getattr(args, "domain", "mixed"))
        if getattr(args, "json", False):
            print(json.dumps(rows, ensure_ascii=False, indent=2,
                             default=str))
        else:
            for row in rows:
                print(f"[{row['reason']}] {row['task']} "
                      f"(priority={row['priority']}, "
                      f"difficulty={row['difficulty']})")
        brain.close()
        return 0
    if command == "math":
        registry = ActuatorRegistry(config)
        operation = getattr(args, "operation", "") or "evaluate"
        result = registry.execute(
            "math", {"operation": operation,
                     "expression": getattr(args, "expression", ""),
                     "variable": getattr(args, "variable", "x")})
        if getattr(args, "json", False):
            print(json.dumps(result.to_dict(), ensure_ascii=False,
                             indent=2, default=str))
        elif result.ok:
            print(result.output)
        else:
            print("ERROR: " + result.error, file=sys.stderr)
            return 1
        brain.close()
        return 0
    print("usage: cognition "
          "{brain|think|observe|act|autonomous|augment|invent|sensor|"
          "curriculum|math|status|grow|tools|capabilities|report}")
    brain.close()
    return 0


def run_cognition_command(args) -> int:
    from cognition.bridge import CognitionRuntime
    from cognition.config import GrowthConfig, default_data_dir

    command = getattr(args, "cognition_command", None) or args.command
    data_dir = os.path.abspath(args.data_dir or default_data_dir())
    kwargs = {"data_dir": data_dir}
    if getattr(args, "dimension", 0):
        kwargs["dimension"] = args.dimension
    if getattr(args, "seed", None) is not None:
        kwargs["seed"] = args.seed
    if getattr(args, "time_budget", None):
        kwargs["time_budget_seconds"] = args.time_budget
    if getattr(args, "max_candidates", None):
        kwargs["max_synthesis_candidates"] = args.max_candidates
    if getattr(args, "network", False):
        kwargs["allow_network"] = True
    config = GrowthConfig(**kwargs)
    if command in ("brain", "think", "observe", "perceive", "act",
                   "autonomous", "augment", "invent", "sensor", "math",
                   "curriculum"):
        return run_rena_command(args, config, command)
    runtime = CognitionRuntime(config)
    if command == "status":
        status = runtime.status()
        if getattr(args, "json", False):
            print(json.dumps(status, ensure_ascii=False, indent=2,
                             default=str))
        else:
            skills = status.get("skills", {})
            tools = status.get("tools", {})
            print(f"DATA: {status.get('data_dir')}")
            print(f"SKILLS: {skills.get('skills', 0)} total, "
                  f"{skills.get('active', 0)} active, "
                  f"{skills.get('candidates', 0)} candidate")
            print(f"TOOLS: {tools.get('counts', {})}")
            print(f"OPEN GOALS: {status.get('open_goal_count', 0)}")
            for name, row in status.get("domains", {}).items():
                print(f"  {name}: active={row['active_skills']} "
                      f"candidate={row['candidate_skills']}")
            for row in status.get("open_goals", [])[-5:]:
                print(f"  goal [{row.get('domain')}] {row.get('text')}")
        return 0
    if command == "grow":
        report = runtime.grow(cycles=args.cycles,
                              domain=args.domain)
        if getattr(args, "json", False):
            print(json.dumps(report.to_dict(), ensure_ascii=False,
                             indent=2, default=str))
        else:
            print(f"GROWTH: {report.tasks_verified}/"
                  f"{report.tasks_attempted} tasks verified, "
                  f"{len(report.skills_active)} active skills, "
                  f"transfer {report.transfer_successes}/"
                  f"{report.transfer_trials}, {report.duration_s}s")
            if report.skills_promoted:
                print("PROMOTED: " + ", ".join(report.skills_promoted))
            for domain, row in report.domains.items():
                if row["attempts"]:
                    print(f"  {domain}: {row['verified']}/"
                          f"{row['attempts']} verified, "
                          f"skills={row['skills_promoted']}")
            print(f"STOPPED: {report.stopped_reason}")
            for limitation in report.limitations[:4]:
                print(f"LIMIT: {limitation}")
        return 0
    if command == "tools":
        summary = runtime.verify_tools(
            include_network=getattr(args, "network", False))
        if getattr(args, "json", False):
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"TOOLS: verified={summary['verified']}")
            if summary["failed"]:
                print(f"       failed={summary['failed']}")
            if summary["skipped"]:
                print(f"       skipped={summary['skipped']}")
        return 0
    if command == "capabilities":
        rows = runtime.capabilities()
        if getattr(args, "json", False):
            print(json.dumps(rows, ensure_ascii=False, indent=2,
                             default=str))
        else:
            if not rows:
                print("No verified cognition capabilities yet. "
                      "Run 'cognition grow' or 'cognition tools'.")
            for row in rows:
                print(f"{row['name']} [{row['kind']}] {row['domain']}: "
                      f"{row['description']}")
        return 0
    if command == "report":
        report = runtime.last_report()
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2,
                             default=str))
        elif report:
            print(f"SESSION: {report.get('session_id')}")
            print(f"TASKS: {report.get('tasks_verified')}/"
                  f"{report.get('tasks_attempted')} verified")
            print(f"SKILLS: {', '.join(report.get('skills_active') or [])}"
                  or "SKILLS: none")
            print(f"TRANSFER: {report.get('transfer_successes')}/"
                  f"{report.get('transfer_trials')}")
            print(f"STOPPED: {report.get('stopped_reason')}")
        else:
            print("No growth report found.")
        return 0
    print("usage: cognition {status|grow|tools|capabilities|report}")
    return 0


def main():
    # Natural Language First: if the first arg is not a known subcommand,
    # treat all args as a natural language request (no technical params).
    KNOWN_COMMANDS = {"generate", "generate-project", "modify", "explain",
                      "learn", "analyze", "verify", "query", "test",
                      "export", "import-data", "computer", "session",
                      "image", "creative", "cognition", "grow"}

    # Extract --debug, --stream, --trace, --voice flags before argparse sees them.
    # This allows:  cli.py --stream=line "some request"
    #                cli.py --stream "some request"
    #                cli.py --debug --stream=function "some request"
    #                cli.py --trace "some request"
    #                cli.py --voice          (real-time voice conversation)
    debug_mode = False
    stream_mode = None  # None = no streaming; "auto"/"line"/"function"/etc.
    trace_mode = False
    voice_mode = False
    # The Think budget: a property of THIS request, extracted before
    # argparse so it works in every invocation form (with or without a
    # subcommand).
    think_requested = False
    deep_think_requested = False
    # The conversation SESSION: a CLI process is one turn, so without
    # it every invocation starts a fresh conversation; with it the same
    # discourse (topic state + record) is loaded before the turn and
    # saved after it.
    session_name = ""

    args_to_keep = []
    i = 1  # skip argv[0]
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--think":
            think_requested = True
            i += 1
        elif arg == "--deep-think":
            deep_think_requested = True
            i += 1
        elif arg == "--debug":
            debug_mode = True
            i += 1
        elif arg == "--trace":
            trace_mode = True
            i += 1
        elif arg == "--voice":
            voice_mode = True
            i += 1
        elif arg == "--session":
            # --session NAME: this turn continues a conversation; the
            # process is one turn of it, not the whole conversation.
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                session_name = sys.argv[i + 1]
                i += 2
            else:
                i += 1
            continue
        elif arg.startswith("--session="):
            session_name = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg.startswith("--stream"):
            # --stream, --stream=line, or --stream line
            if "=" in arg:
                stream_mode = arg.split("=", 1)[1] or "auto"
            elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                stream_mode = sys.argv[i + 1]
                i += 2
                continue
            else:
                stream_mode = "auto"
            i += 1
        else:
            args_to_keep.append(arg)
            i += 1

    # Rebuild sys.argv for any downstream code that still reads it
    sys.argv = [sys.argv[0]] + args_to_keep

    if len(args_to_keep) > 0 and args_to_keep[0] not in KNOWN_COMMANDS and \
       not args_to_keep[0].startswith("-"):
        # Natural language mode: join all remaining args as a single request
        request = " ".join(args_to_keep)
        # The request's reasoning budget: read by the whole pipeline
        # (ThinkEngine hypotheses, alternative plans, verification
        # inputs, repair attempts) through the budget context.
        from brain.think_budget import set_think_mode, current_budget, resolve_mode
        set_think_mode(resolve_mode(think=think_requested,
                                    deep_think=deep_think_requested))
        _print_think_header(current_budget())
        brain = CodeBrain()
        if session_name:
            _load_session(session_name, brain)
        try:
            if trace_mode:
                trace_algorithmic(brain, request)
            elif debug_mode:
                result = brain.process_natural_debug(request)
                print_debug_result(result)
            elif stream_mode:
                result = brain.process_with_streaming(
                    request, stream_mode=stream_mode)
                print_streaming_result(result)
            else:
                result = brain.process_natural(request)
                print_natural_result(result)
        finally:
            # The conversation continues past this process: the state
            # is saved under the session it belongs to (a one-shot
            # turn without --session saves nothing).
            if session_name:
                _save_session(session_name, brain)
        return

    if voice_mode:
        run_voice_conversation()
        return

    parser = argparse.ArgumentParser(
        prog="RENA-BR 0.5 Beta",
        description="RENA-BR - AI Code Generation without Transformers",
    )
    parser.add_argument("--version", action="version", version="RENA-BR 0.5 Beta")
    # The Think budget as a TOP-LEVEL option (also accepted by the
    # generate/generate-project subcommands after the request).
    parser.add_argument("--think", dest="think_top", action="store_true",
                        help="Think: a larger reasoning budget — more candidate "
                             "hypotheses, alternative plans, adversarial "
                             "verification inputs, and more repair attempts")
    parser.add_argument("--deep-think", dest="deep_think_top", action="store_true",
                        help="Deep Think: the largest reasoning budget — the full "
                             "candidate space, the most verification and the "
                             "most repair iterations")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    gen_parser = subparsers.add_parser("generate", help="Generate code from natural language")
    gen_parser.add_argument("request", help="Natural language request")
    gen_parser.add_argument("--lang", "-l", default="python", help="Target language")
    gen_parser.add_argument("--all", "-a", action="store_true", help="Generate for all languages")
    gen_parser.add_argument("--ir", action="store_true", help="Show IR tree before code")
    gen_parser.add_argument("--ir-json", action="store_true", help="Show IR as JSON")
    gen_parser.add_argument("--think", action="store_true",
                            help="Think: a larger reasoning budget — more "
                                 "candidate hypotheses, alternative plans, "
                                 "and verification inputs")
    gen_parser.add_argument("--deep-think", action="store_true",
                            help="Deep Think: the largest reasoning budget — "
                                 "the full candidate space, adversarial "
                                 "inputs, and more repair attempts")

    proj_parser = subparsers.add_parser("generate-project", help="Generate a full project from natural language")
    proj_parser.add_argument("request", help="Natural language request for the project")
    proj_parser.add_argument("--lang", "-l", default="kotlin", help="Primary language (default: kotlin)")
    proj_parser.add_argument("--output", "-o", help="Output directory for generated files")
    proj_parser.add_argument("--ir", action="store_true", help="Show Project IR tree")
    proj_parser.add_argument("--ir-json", action="store_true", help="Show Project IR as JSON")
    proj_parser.add_argument("--think", action="store_true",
                             help="Think: a larger reasoning budget")
    proj_parser.add_argument("--deep-think", action="store_true",
                             help="Deep Think: the largest reasoning budget")

    modify_parser = subparsers.add_parser("modify", help="Modify existing IR")
    modify_parser.add_argument("action", choices=["add", "modify", "remove", "move", "replace", "set_property"])
    modify_parser.add_argument("--target", "-t", required=True, help="Target node ID")
    modify_parser.add_argument("--name", "-n", help="Node name")
    modify_parser.add_argument("--type", dest="node_type", help="Node type")
    modify_parser.add_argument("--value", "-v", help="Node value")
    modify_parser.add_argument("--ir", help="IR JSON file")

    explain_parser = subparsers.add_parser("explain", help="Explain reasoning process")
    explain_parser.add_argument("request", help="Natural language request")
    explain_parser.add_argument("--lang", "-l", default="python", help="Target language")

    learn_parser = subparsers.add_parser("learn", help="Learn from source file")
    learn_parser.add_argument("file", help="Source file to learn from")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze code")
    analyze_parser.add_argument("file", help="Source file to analyze")

    verify_parser = subparsers.add_parser("verify", help="Verify generated code")
    verify_parser.add_argument("code", help="Code string to verify")
    verify_parser.add_argument("--lang", "-l", required=True, help="Language")

    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("type", choices=["concepts", "relations", "memory"])
    query_parser.add_argument("query_text", nargs="?", default="")
    query_parser.add_argument("--lang", "-l", default="")

    test_parser = subparsers.add_parser("test", help="Run end-to-end tests")
    test_parser.add_argument("--verbose", "-v", action="store_true")

    image_parser = subparsers.add_parser(
        "image", help="Image input/output: describe, convert, chart")
    image_sub = image_parser.add_subparsers(dest="image_command")
    img_describe = image_sub.add_parser(
        "describe", help="Describe an image: regions, palette, histogram")
    img_describe.add_argument("path", help="image file (PNG/BMP/PPM)")
    img_describe.add_argument("--json", action="store_true",
                              help="print the structured description as JSON")
    img_convert = image_sub.add_parser(
        "convert", help="Apply an operation and save the result")
    img_convert.add_argument("input")
    img_convert.add_argument("output")
    img_convert.add_argument("--op", default="gray",
                             help="gray|invert|blur|edges|threshold|resize|brighten|crop")
    img_convert.add_argument("--radius", type=int, default=2)
    img_convert.add_argument("--level", type=int, default=128)
    img_convert.add_argument("--width", type=int, default=0)
    img_convert.add_argument("--height", type=int, default=0)
    img_convert.add_argument("--delta", type=int, default=40)
    img_convert.add_argument("--box", type=int, nargs=4, default=None,
                             help="crop box: x y width height")
    img_render = image_sub.add_parser(
        "render",
        help="Physics-based (ray traced) photorealistic scene rendering")
    img_render.add_argument("request",
                            help='scene name or request '
                                 '("space_planet", "사실적인 우주 행성")')
    img_render.add_argument("output", nargs="?", default="")
    img_render.add_argument("--scene", default="",
                            help="explicit scene name")
    img_render.add_argument("--width", type=int, default=400)
    img_render.add_argument("--height", type=int, default=300)
    img_render.add_argument("--samples", type=int, default=6)
    img_render.add_argument("--bounces", type=int, default=0,
                            help="0 = scene default")
    img_render.add_argument("--seed", type=int, default=0)
    img_draw = image_sub.add_parser(
        "draw", help="Draw a scene from natural language (icon composition)")
    img_draw.add_argument("request",
                          help='e.g. "사람 그려줘" / "draw a person and a bed"')
    img_draw.add_argument("output", nargs="?", default="",
                          help="output image path (default: ~/.rena_brain/images/)")
    img_draw.add_argument("--width", type=int, default=640)
    img_draw.add_argument("--height", type=int, default=480)
    img_draw.add_argument("--engine", default="auto",
                          choices=["auto", "icons", "ppgen"],
                          help="auto: icon registry first, then the "
                               "parametric part-morphology generator")
    img_draw.add_argument("--seed", type=int, default=0)
    img_chart = image_sub.add_parser(
        "chart", help="Render a bar/line chart from JSON data")
    img_chart.add_argument("input", help="JSON data file")
    img_chart.add_argument("output", help="output image path (.png)")
    img_chart.add_argument("--kind", default="bar", help="bar|line")
    img_chart.add_argument("--title", default="")

    creative_parser = subparsers.add_parser(
        "creative",
        help="Creative engine: plan a World IR, render it (video/audio "
             "land in later phases)")
    creative_sub = creative_parser.add_subparsers(dest="creative_command")
    c_plan = creative_sub.add_parser(
        "plan", help="Plan a World IR from a description")
    c_plan.add_argument("request")
    c_plan.add_argument("--seed", type=int, default=0)
    c_plan.add_argument("--duration", type=float, default=0.0)
    c_plan.add_argument("--output", default="",
                        help="write the World IR JSON to this path")
    c_plan.add_argument("--json", action="store_true",
                        help="print the full World IR JSON")
    c_img = creative_sub.add_parser(
        "image", help="Render a physics-based image from a description")
    c_img.add_argument("request")
    c_img.add_argument("--output", default="")
    c_img.add_argument("--width", type=int, default=480)
    c_img.add_argument("--height", type=int, default=360)
    c_img.add_argument("--samples", type=int, default=4)
    c_img.add_argument("--seed", type=int, default=0)
    c_img.add_argument("--duration", type=float, default=0.0)
    c_video = creative_sub.add_parser(
        "video",
        help="Render a timed video (MJPEG AVI — no ffmpeg in this env)")
    c_video.add_argument("request")
    c_video.add_argument("--output", default="")
    c_video.add_argument("--duration", type=float, default=6.0)
    c_video.add_argument("--fps", type=float, default=12.0)
    c_video.add_argument("--width", type=int, default=192)
    c_video.add_argument("--height", type=int, default=144)
    c_video.add_argument("--samples", type=int, default=2)
    c_video.add_argument("--seed", type=int, default=0)
    c_video.add_argument("--frames-dir", default="",
                         help="also dump PNG frames here")
    c_video.add_argument("--frames-every", type=int, default=0)
    c_inspect = creative_sub.add_parser(
        "inspect-video",
        help="Read-only video analysis (MJPEG AVI or image sequence)")
    c_inspect.add_argument("input")
    c_inspect.add_argument("--output", default="",
                           help="write the analysis JSON here")
    c_inspect.add_argument("--max-frames", type=int, default=120)
    c_inspect.add_argument("--threshold", type=float, default=0.18)
    c_audio = creative_sub.add_parser(
        "audio", help="Synthesize music or effects (WAV, procedural)")
    c_audio.add_argument("--kind", choices=["music", "effects", "speech"],
                         required=True)
    c_audio.add_argument("--text", default="",
                         help="speech text (--kind speech)")
    c_audio.add_argument("--style", default="ambient",
                         help="music style: ambient|mystical|future|warm")
    c_audio.add_argument("--prompt", default="",
                         help="effect prompt, e.g. '신비로운 유리 종소리'")
    c_audio.add_argument("--duration", type=float, default=8.0)
    c_audio.add_argument("--seed", type=int, default=0)
    c_audio.add_argument("--output", default="")
    c_audio.add_argument("--rate", type=int, default=22050)
    c_compose = creative_sub.add_parser(
        "compose", help="Render video + synthesized audio into one AVI")
    c_compose.add_argument("request")
    c_compose.add_argument("--output", default="")
    c_compose.add_argument("--duration", type=float, default=6.0)
    c_compose.add_argument("--fps", type=float, default=12.0)
    c_compose.add_argument("--width", type=int, default=160)
    c_compose.add_argument("--height", type=int, default=120)
    c_compose.add_argument("--samples", type=int, default=1)
    c_compose.add_argument("--seed", type=int, default=0)
    c_compose.add_argument("--audio", default="music,effects",
                           help="music,effects or empty for silent")
    c_bench = creative_sub.add_parser(
        "benchmark", help="Measured CPU/GPU capability benchmark")
    c_bench.add_argument("--device", default="auto",
                         choices=["auto", "cpu", "gpu"])
    c_bench.add_argument("--output", default="output/creative/benchmark.json")
    c_eval = creative_sub.add_parser(
        "evaluate", help="Evaluate generated artifacts in a directory")
    c_eval.add_argument("directory", nargs="?", default="output/creative")
    c_eval.add_argument("--output", default="")
    c_learn = creative_sub.add_parser(
        "learn", help="Tiny-model experiment vs honest baselines")
    c_learn.add_argument("--output", default="output/creative/learning/experiment.json")

    computer_parser = subparsers.add_parser("computer", help="Run the General Computer Agent on a natural-language desktop task")
    computer_parser.add_argument("request", help="Natural language task, e.g. 'Android Studio에서 프로젝트를 열고 빌드해줘'")
    computer_parser.add_argument("--confirm", action="store_true", help="Allow high-risk actions with an interactive confirmation prompt")
    computer_parser.add_argument("--trace", "-t", action="store_true", help="Print the full agent loop trace")
    computer_parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    computer_parser.add_argument("--backend", "-b", default="auto", choices=["auto", "windows", "android", "mock"], help="Target platform backend (android = ADB device/emulator)")
    computer_parser.add_argument("--device", "-d", default=None, help="ADB device id for the android backend")

    export_parser = subparsers.add_parser("export", help="Export brain data")
    export_parser.add_argument("--output", "-o", default="brain_data.json")

    import_parser = subparsers.add_parser("import-data", help="Import brain data")
    import_parser.add_argument("input_file", help="JSON file to import")

    session_parser = subparsers.add_parser(
        "session",
        help="Continue a conversation session — a CLI process is one "
             "turn of it, not the conversation")
    session_parser.add_argument("name", help="Session name, e.g. mychat")
    session_parser.add_argument("--turn", default="",
                                help="One turn in this session, then exit")
    session_parser.add_argument("--new", action="store_true",
                                help="Start this session from scratch")

    def _add_growth_args(target):
        target.add_argument("--cycles", type=int, default=6,
                            help="bounded growth cycles (default: 6)")
        target.add_argument("--domain", default="",
                            help="practice one domain (default: all)")
        target.add_argument("--seed", type=int, default=13)
        target.add_argument("--dimension", type=int, default=0,
                            help="VSA dimension (default: config 2048)")
        target.add_argument("--time-budget", type=float, default=120.0,
                            help="wall-clock seconds for the growth run")
        target.add_argument("--max-candidates", type=int, default=20000)
        target.add_argument("--data-dir", default="",
                            help="cognition data directory")
        target.add_argument("--json", action="store_true",
                            help="machine-readable JSON output")

    grow_parser = subparsers.add_parser(
        "grow", help="Run the autonomous failure -> skill growth loop")
    _add_growth_args(grow_parser)

    cognition_parser = subparsers.add_parser(
        "cognition", help="Inspect or run the cognitive growth subsystem")
    cognition_sub = cognition_parser.add_subparsers(
        dest="cognition_command")
    cog_status = cognition_sub.add_parser(
        "status", help="Show skills, goals, memory and tool status")
    cog_status.add_argument("--data-dir", default="")
    cog_status.add_argument("--json", action="store_true")
    cog_grow = cognition_sub.add_parser(
        "grow", help="Run the growth loop")
    _add_growth_args(cog_grow)
    cog_tools = cognition_sub.add_parser(
        "tools", help="Verify web/video/image/data tools before registering")
    cog_tools.add_argument("--data-dir", default="")
    cog_tools.add_argument("--network", action="store_true",
                           help="allow the external web probe")
    cog_tools.add_argument("--json", action="store_true")
    cog_caps = cognition_sub.add_parser(
        "capabilities", help="List active verified capabilities")
    cog_caps.add_argument("--data-dir", default="")
    cog_caps.add_argument("--json", action="store_true")
    cog_report = cognition_sub.add_parser(
        "report", help="Show the last growth report")
    cog_report.add_argument("--data-dir", default="")
    cog_report.add_argument("--json", action="store_true")

    cog_brain = cognition_sub.add_parser(
        "brain", help="Show unified RenaBrain status")
    cog_brain.add_argument("--data-dir", default="")
    cog_brain.add_argument("--dimension", type=int, default=0)
    cog_brain.add_argument("--json", action="store_true")
    cog_think = cognition_sub.add_parser(
        "think", help="Retrieve concepts and predict from a query")
    cog_think.add_argument("query")
    cog_think.add_argument("--modality", default="text")
    cog_think.add_argument("--data-dir", default="")
    cog_think.add_argument("--dimension", type=int, default=0)
    cog_think.add_argument("--json", action="store_true")
    cog_observe = cognition_sub.add_parser(
        "observe", help="Encode and store a perception")
    cog_observe.add_argument("--modality", default="text")
    cog_observe.add_argument("--text", default="")
    cog_observe.add_argument("--path", default="")
    cog_observe.add_argument("--label", default="")
    cog_observe.add_argument("--data-dir", default="")
    cog_observe.add_argument("--dimension", type=int, default=0)
    cog_observe.add_argument("--json", action="store_true")
    cog_act = cognition_sub.add_parser(
        "act", help="Call a capability actuator")
    cog_act.add_argument("actuator",
                         choices=["math", "code", "image", "video",
                                  "audio", "web", "file", "data"])
    cog_act.add_argument("--operation", default="")
    cog_act.add_argument("--code", default="")
    cog_act.add_argument("--expression", default="")
    cog_act.add_argument("--path", default="")
    cog_act.add_argument("--prompt", default="")
    cog_act.add_argument("--request", default="")
    cog_act.add_argument("--force", action="store_true")
    cog_act.add_argument("--network", action="store_true")
    cog_act.add_argument("--data-dir", default="")
    cog_act.add_argument("--dimension", type=int, default=0)
    cog_act.add_argument("--json", action="store_true")
    cog_auto = cognition_sub.add_parser(
        "autonomous", help="Bounded autonomous development loop")
    cog_auto.add_argument("--cycles", type=int, default=4)
    cog_auto.add_argument("--time-budget", type=float, default=120.0)
    cog_auto.add_argument("--data-dir", default="")
    cog_auto.add_argument("--dimension", type=int, default=0)
    cog_auto.add_argument("--json", action="store_true")
    cog_augment = cognition_sub.add_parser(
        "augment", help="Self-augment: compose and verify new skills")
    cog_augment.add_argument("--cycles", type=int, default=1)
    cog_augment.add_argument("--candidates", type=int, default=6)
    cog_augment.add_argument("--trials", type=int, default=0,
                             help="fresh verification trials (0=config)")
    cog_augment.add_argument("--domain", default="math")
    cog_augment.add_argument("--time-budget", type=float, default=20.0)
    cog_augment.add_argument("--data-dir", default="")
    cog_augment.add_argument("--dimension", type=int, default=0)
    cog_augment.add_argument("--json", action="store_true")
    cog_invent = cognition_sub.add_parser(
        "invent", help="Invent a primitive or a virtual sensor")
    cog_invent.add_argument("--kind", choices=["primitive", "sensor"],
                            default="primitive")
    cog_invent.add_argument("--modality", default="text")
    cog_invent.add_argument("--sample", action="append", default=[],
                            help="sample source for sensor invention")
    cog_invent.add_argument("--cycles", type=int, default=1)
    cog_invent.add_argument("--candidates", type=int, default=0)
    cog_invent.add_argument("--time-budget", type=float, default=20.0)
    cog_invent.add_argument("--data-dir", default="")
    cog_invent.add_argument("--dimension", type=int, default=0)
    cog_invent.add_argument("--json", action="store_true")
    cog_sensor = cognition_sub.add_parser(
        "sensor", help="List, read or invent virtual sensors")
    cog_sensor.add_argument("sensor_action",
                            choices=["list", "read", "invent"],
                            default="list", nargs="?")
    cog_sensor.add_argument("--name", default="")
    cog_sensor.add_argument("--source", default="")
    cog_sensor.add_argument("--modality", default="text")
    cog_sensor.add_argument("--sample", action="append", default=[])
    cog_sensor.add_argument("--cycles", type=int, default=1)
    cog_sensor.add_argument("--candidates", type=int, default=12)
    cog_sensor.add_argument("--time-budget", type=float, default=20.0)
    cog_sensor.add_argument("--data-dir", default="")
    cog_sensor.add_argument("--dimension", type=int, default=0)
    cog_sensor.add_argument("--json", action="store_true")
    cog_curriculum = cognition_sub.add_parser(
        "curriculum", help="Propose a metacognitive practice curriculum")
    cog_curriculum.add_argument("--count", type=int, default=6)
    cog_curriculum.add_argument("--domain", default="mixed")
    cog_curriculum.add_argument("--data-dir", default="")
    cog_curriculum.add_argument("--dimension", type=int, default=0)
    cog_curriculum.add_argument("--json", action="store_true")
    cog_math = cognition_sub.add_parser(
        "math", help="Evaluate, differentiate or solve an expression")
    cog_math.add_argument("expression")
    cog_math.add_argument("--operation", default="evaluate",
                          choices=["evaluate", "differentiate",
                                   "simplify", "solve", "variables"])
    cog_math.add_argument("--variable", default="x")
    cog_math.add_argument("--data-dir", default="")
    cog_math.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # The think budget: --deep-think wins over --think; the mode is a
    # property of THIS request, read by the whole pipeline through the
    # budget context (ThinkEngine, generation controller, repair). The
    # flags may come from the TOP-LEVEL parser or the subcommand — both
    # mean the same request.
    from brain.think_budget import set_think_mode, current_budget, resolve_mode
    mode = resolve_mode(
        getattr(args, "mode", None),
        think=bool(getattr(args, "think", False)
                   or getattr(args, "think_top", False)
                   or think_requested),
        deep_think=bool(getattr(args, "deep_think", False)
                        or getattr(args, "deep_think_top", False)
                        or deep_think_requested),
    )
    set_think_mode(mode)
    if args.command in ("generate", "generate-project"):
        _print_think_header(current_budget())

    if args.command in ("grow", "cognition"):
        return run_cognition_command(args)

    brain = CodeBrain()

    if args.command == "session":
        # The conversation continues across processes: the session's
        # topic state and turn record are loaded, this turn runs, and
        # the state is saved for the next one.
        run_session(brain, args.name, turn=args.turn, fresh=args.new)
        return

    if args.command == "generate":
        result = brain.process(args.request, args.lang)
        print(f"\n=== 요청: {result.get('request', args.request)} ===")
        print(f"=== 언어: {result.get('language', args.lang)} ===")
        _conf = result.get("confidence")
        if isinstance(_conf, dict) and "confidence" in _conf:
            print(f"=== 신뢰도: {_conf.get('confidence')} "
                  f"({_conf.get('basis')}) ===")
        if current_budget().mode.value != "normal":
            reasoning = result.get("reasoning")
            n_hyps = len(getattr(reasoning, "hypotheses", []) or []) \
                if reasoning is not None else 0
            n_alts = len(getattr(reasoning, "alternative_plans", []) or []) \
                if reasoning is not None else 0
            print(f"=== 추론 모드: {current_budget().mode.value} "
                  f"(후보 {n_hyps}, 대안 계획 {n_alts}) ===")

        # Handle both legacy (reasoning key) and semantic pipeline formats
        if "reasoning" in result:
            reasoning = result["reasoning"]
            if hasattr(reasoning, "explanation"):
                print(f"\n--- 추론 ---")
                print(reasoning.explanation)
            if hasattr(reasoning, "plan") and reasoning.plan:
                print(f"\n--- 계획 ---")
                for action in reasoning.plan:
                    if hasattr(action, "kind"):
                        print(f"  [{action.kind}] {getattr(action, 'reason', '')}")
                    elif isinstance(action, dict):
                        print(f"  [{action.get('action_type', '')}] {action.get('description', '')}")
        elif "semantic" in result:
            semantic = result["semantic"] if isinstance(result["semantic"], dict) else {}
            goal = semantic.get("goal", {})
            ops = [op.get("name", "") for op in semantic.get("operations", [])]
            print(f"\n--- 분석 ---")
            print(f"  목표: {goal.get('goal_type', 'unknown')}")
            if ops:
                print(f"  연산: {ops}")

        print(f"\n--- 생성된 코드 ---")
        for name, data in result.get("results", {}).items():
            if getattr(args, 'ir', False) or getattr(args, 'ir_json', False):
                print(f"\n--- IR: {name} ---")
                ir_data = data.get("ir", {})
                if isinstance(ir_data, dict) and "type" in ir_data:
                    print(json.dumps(ir_data, indent=2, ensure_ascii=False))
                else:
                    try:
                        ir_node = IRNode.from_dict(ir_data)
                        if getattr(args, 'ir_json', False):
                            print(json.dumps(ir_data, indent=2, ensure_ascii=False))
                        else:
                            print(ir_node.pretty_print())
                    except Exception:
                        print(json.dumps(ir_data, indent=2, ensure_ascii=False))
            if args.all:
                for lang in _ordered_languages(data.get("languages", {}),
                                               args.lang):
                    lang_data = data.get("languages", {}).get(lang, {})
                    print(f"\n# {lang.upper()}")
                    print(lang_data.get("code", ""))
            else:
                lang_data = data.get("languages", {}).get(args.lang, {})
                print(f"\n# {args.lang.upper()}")
                print(lang_data.get("code", "Not available"))

    elif args.command == "generate-project":
        intent_result = brain.intent_analyzer.analyze(args.request)

        if intent_result.intent == "CREATE_PROJECT":
            result = brain.generate_project(args.request, args.lang, args.output)
            print(f"\n=== 프로젝트 생성: {result['request']} ===")
            print(f"=== 언어: {result['language']} ===")
            if 'technology' in result:
                tech = result['technology']
                print(f"=== 플랫폼: {tech.get('platform', 'N/A')} ===")
                print(f"=== 프레임워크: {tech.get('framework', 'N/A')} ===")
            if getattr(args, 'ir', False) or getattr(args, 'ir_json', False):
                print(f"\n--- Project IR ---")
                if getattr(args, 'ir_json', False):
                    print(json.dumps(result["project_ir"], indent=2, ensure_ascii=False))
                else:
                    from code.ir.project_ir import ProjectIR
                    proj_ir = ProjectIR.from_dict(result["project_ir"])
                    print(proj_ir.pretty_print())
            print(f"\n--- 생성된 파일 ({len(result['files'])}개) ---")
            for f in result["files"]:
                print(f"\n  [{f['file_type']}] {f['path']}")
                print(f"  {'─' * 40}")
                content = f['content']
                for line in content.split('\n')[:20]:
                    print(f"    {line}")
                if content.count('\n') > 20:
                    print(f"    ... ({content.count(chr(10))} total lines)")
            print(f"\n--- 검증 ---")
            v = result["verification"]
            status = "PASS" if v["success"] else "FAIL"
            print(f"  Status: {status} (passed: {v['checks_passed']}, failed: {v['checks_failed']})")
            if v["issues"]:
                print(f"  Issues:")
                for issue in v["issues"]:
                    sev = issue["severity"].upper()
                    print(f"    [{sev}] {issue['category']}: {issue['message']}")
                    if issue.get("suggestion"):
                        print(f"      suggestion: {issue['suggestion']}")
            if result.get("error_analyses"):
                print(f"\n--- Error Analysis ---")
                for ea in result["error_analyses"]:
                    print(f"  Type: {ea['error_type']}")
                    print(f"  Root cause: {ea['root_cause']}")
                    for s in ea.get("suggestions", []):
                        print(f"  Suggestion: {s}")
            if args.output:
                print(f"\n--- 출력 ---")
                print(f"  프로젝트 파일이 {args.output}/ 에 저장되었습니다.")
        elif intent_result.intent == "GENERATE_CODE":
            result = brain.process(args.request, args.lang)
            print(f"\n=== 코드 생성: {result['request']} ===")
            print(f"=== 언어: {result['language']} ===")
            for name, data in result["results"].items():
                print(f"\n--- {name} ---")
                for lang in _ordered_languages(data["languages"], args.lang):
                    lang_data = data["languages"][lang]
                    print(f"\n# {lang.upper()}")
                    print(lang_data["code"])
        else:
            if intent_result.intent == "CHAT":
                response = brain.intent_analyzer.get_chat_response(args.request)
            elif intent_result.intent == "QUESTION":
                response = brain.intent_analyzer.get_question_response(args.request)
            elif intent_result.intent == "EXPLANATION":
                response = brain.intent_analyzer.get_explanation_response(args.request)
            else:
                response = brain.intent_analyzer.get_unknown_response(args.request)
            print(f"\n[Intent] {intent_result.intent}")
            print(f"[Response]\n{response}")
            print(f"\n[Project] No project created.")

    elif args.command == "explain":
        explanation = brain.explain(args.request, args.lang)
        print(explanation)

    elif args.command == "learn":
        result = brain.learn_file(args.file)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "analyze":
        result = brain.learn_file(args.file)
        print(f"File: {result['file']}")
        print(f"Language: {result['language']}")
        print(f"Result: {json.dumps(result['result'], indent=2)}")

    elif args.command == "verify":
        result = brain.verify(args.code, args.lang)
        if result.success:
            status = "PASS"
        else:
            missing_tool = any("not installed" in e.lower() or "No such file" in e for e in result.errors)
            status = "TOOLCHAIN NOT INSTALLED" if missing_tool else "FAIL"
        print(f"Verification: {status}")
        if result.errors:
            for e in result.errors:
                print(f"  ERROR: {e}")
        if result.output:
            print(f"  Output: {result.output}")

    elif args.command == "query":
        if args.type == "concepts":
            results = brain.query_concepts(args.query_text)
            for c in results:
                print(f"[{c.id}] {c.name} ({c.category}) - langs: {c.languages}")
        elif args.type == "relations":
            results = brain.query_relations()
            for r in results:
                print(f"[{r.id}] {r.source_id} --{r.relation_type.value}--> {r.target_id}")
        elif args.type == "memory":
            results = brain.query_memory(args.query_text, args.lang)
            for m in results:
                print(f"[{m.id}] {m.memory_type.value} ({m.language}) - use_count: {m.use_count}")

    elif args.command == "test":
        run_e2e_test(brain, verbose=args.verbose)

    elif args.command == "export":
        data = brain.export_data()
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data exported to {args.output}")

    elif args.command == "import-data":
        with open(args.input_file) as f:
            data = json.load(f)
        result = brain.import_data(data)
        print(f"Import results: {result}")

    elif args.command == "image":
        from brain import image_io
        try:
            if args.image_command == "describe":
                desc = image_io.describe_image(args.path)
                if getattr(args, "json", False):
                    print(json.dumps(desc, ensure_ascii=False, indent=2))
                else:
                    print(desc["summary"])
                    for region in desc["regions"][:8]:
                        print(f"  {region['shape']} bbox={region['bbox']} "
                              f"area={region['area']} "
                              f"mean_rgb={tuple(region['mean_color'])}")
            elif args.image_command == "convert":
                out = image_io.convert_image(
                    args.input, args.output, args.op,
                    radius=args.radius, level=args.level,
                    width=args.width, height=args.height,
                    delta=args.delta, crop_box=args.box)
                print(f"[image] {args.op}: {args.input} -> {out}")
            elif args.image_command == "render":
                from brain import render3d
                scene_name = args.scene or ""
                if not scene_name and args.request in render3d.SCENES:
                    scene_name = args.request
                if not scene_name:
                    scene_name = render3d.parse_render_request(
                        args.request) or ""
                if not scene_name:
                    print("[image] render: unknown scene. available: "
                          + ", ".join(render3d.scene_names()))
                    return 1
                out = args.output or render3d.default_render_path(scene_name)
                meta = render3d.render_to_file(
                    scene_name, out, args.width, args.height,
                    args.samples,
                    seed=(args.seed or None),
                    bounces=(args.bounces or None))
                print(f"[image] render: {meta['summary']} -> {meta['path']}")
            elif args.image_command == "draw":
                engine = getattr(args, "engine", "auto")
                # 1) the existing icon registry keeps its behavior for
                #    requests it can draw (never replaced unconditionally)
                if engine in ("auto", "icons"):
                    from brain.image_draw import (
                        parse_draw_request, draw_request,
                        default_output_path)
                    icons = parse_draw_request(args.request)
                    if engine == "icons" or icons:
                        try:
                            out = args.output or default_output_path(
                                args.request)
                            meta = draw_request(args.request, out,
                                                args.width, args.height)
                            print(f"[image] draw (icons): "
                                  f"{meta['summary']} -> {meta['path']}")
                            return 0
                        except Exception as exc:
                            print(f"[image] error: {exc}")
                            return 1
                # 2) parametric part-morphology generator: any noun,
                #    modifier and relation becomes a part assembly
                if engine == "icons":
                    print("[image] error: icon engine found no drawable "
                          "object in this request")
                    return 1
                try:
                    from research.render import generate as ppgen_generate
                    meta = ppgen_generate(
                        args.request, args.width, args.height,
                        seed=getattr(args, "seed", 0),
                        output_path=args.output or "")
                except Exception as exc:
                    print(f"[image] error: {exc}")
                    return 1
                print(f"[image] draw (ppgen): {meta['path']} "
                      f"({meta['width']}x{meta['height']}, "
                      f"{meta['seconds']}s, parts={meta['parts']}, "
                      f"coverage={meta['constraints']['coverage']})")
                print("  entities: " + ", ".join(
                    f"{name}[{family}{'' if src == 'cue' else '/' + src}]"
                    f"x{count}"
                    for name, family, src, count in meta["entities"]))
                if meta["relations"]:
                    print("  relations: " + ", ".join(
                        f"{a}-{rel}-{b}" for a, rel, b
                        in meta["relations"]))
                if meta["assumptions"]:
                    for note in meta["assumptions"]:
                        print(f"  assumption: {note}")
                if meta["unsupported"]:
                    for note in meta["unsupported"]:
                        print(f"  unsupported: {note}")
                if meta["constraints"]["repairs"]:
                    print("  repairs: " + "; ".join(
                        meta["constraints"]["repairs"]))
                return 0
            elif args.image_command == "chart":
                out = image_io.chart_from_json(
                    args.input, args.output,
                    kind=args.kind, title=args.title)
                print(f"[image] chart({args.kind}): {args.input} -> {out}")
            else:
                image_parser.print_help()
        except image_io.ImageError as exc:
            print(f"[image] error: {exc}")
            return 1
    elif args.command == "creative":
        if args.creative_command == "plan":
            from creative.planner import (plan_world,
                                          UnsupportedCreativeRequest)
            try:
                world = plan_world(args.request, seed=args.seed,
                                   duration=args.duration or None)
            except UnsupportedCreativeRequest as exc:
                print(f"[creative] unsupported request: {exc}")
                return 1
            problems = world.validate()
            if args.output:
                from media.image.generate import resolve_output_path
                out = resolve_output_path(args.output)
                with open(out, "w", encoding="utf-8") as handle:
                    handle.write(world.to_json())
                print(f"[creative] world written: {out}")
            if args.json:
                print(world.to_json())
            else:
                score = world.provenance.get("score", {})
                print(f"[creative] plan '{world.name}' "
                      f"(seed={world.seed}, objects={len(world.objects)}, "
                      f"score={score.get('score')}, "
                      f"coverage={score.get('coverage')})")
                print("  objects: " + ", ".join(
                    f"{o.id}({o.material.get('shade')})"
                    for o in world.objects))
                print("  environment: "
                      f"{world.environment.get('kind')} | "
                      f"lights: {len(world.lights)} | "
                      f"events: {len(world.events)} | "
                      f"audio: {len(world.audio)}")
                print("  rules: " + ", ".join(
                    world.provenance.get("rules", [])[:10]))
                if problems:
                    print("  problems: " + "; ".join(problems[:5]))
            return 0
        if args.creative_command == "image":
            from media.image.generate import generate_image
            from creative.planner import UnsupportedCreativeRequest
            try:
                meta = generate_image(
                    args.request, output_path=args.output,
                    width=args.width, height=args.height,
                    samples=args.samples, seed=args.seed,
                    duration=args.duration)
            except UnsupportedCreativeRequest as exc:
                print(f"[creative] unsupported request: {exc}")
                return 1
            except Exception as exc:
                print(f"[creative] image error: {exc}")
                return 1
            print(f"[creative] image: {meta['path']} "
                  f"({meta['width']}x{meta['height']}, "
                  f"{meta['samples']}spp, {meta['render_seconds']}s)")
            print("  world: " + ", ".join(meta["world"]["objects"])
                  + f" | env={meta['world']['environment']}")
            score = meta["provenance"].get("score", {})
            print(f"  score={score.get('score')} "
                  f"coverage={score.get('coverage')} "
                  f"candidates={meta['provenance'].get('candidates')}")
            return 0
        if args.creative_command == "video":
            from media.video.generate import generate_video
            from creative.planner import UnsupportedCreativeRequest
            try:
                meta = generate_video(
                    args.request, output_path=args.output,
                    duration=args.duration, fps=args.fps,
                    width=args.width, height=args.height,
                    samples=args.samples, seed=args.seed,
                    frames_dir=args.frames_dir,
                    frames_every=args.frames_every)
            except UnsupportedCreativeRequest as exc:
                print(f"[creative] unsupported request: {exc}")
                return 1
            except Exception as exc:
                print(f"[creative] video error: {exc}")
                return 1
            print(f"[creative] video: {meta['path']} "
                  f"({meta['width']}x{meta['height']}, {meta['fps']}fps, "
                  f"{meta['frame_count']} frames, {meta['duration']}s, "
                  f"{meta['bytes'] // 1024}KB, "
                  f"{meta['render_encode_seconds']}s)")
            print("  moving: " + (", ".join(meta["moving_objects"])
                                  or "(none)"))
            if meta.get("frames_dumped"):
                print(f"  frames dumped: {meta['frames_dumped']}")
            return 0
        if args.creative_command == "audio":
            from media.audio import synthesis as syn
            from media.audio.wav import check_signal, validate_wav, write_wav
            from media.image.generate import (DEFAULT_OUT_DIR,
                                              resolve_output_path)
            import time as _time
            try:
                if args.kind == "speech":
                    from media.audio.formant import speak_text
                    text = args.text or args.prompt
                    if not text:
                        print("[creative] audio error: --kind speech "
                              "requires --text")
                        return 1
                    signal, phonemes = speak_text(text)
                    label = f"speech:{text[:24]}"
                    spec_extra = {"phonemes": len(phonemes)}
                elif args.kind == "music":
                    signal = syn.generate_music(args.style, args.duration,
                                                seed=args.seed,
                                                sample_rate=args.rate)
                    label = f"music:{args.style}"
                else:
                    effect = syn.effect_from_prompt(args.prompt)
                    signal = syn.generate_effect(effect, args.duration,
                                                 seed=args.seed,
                                                 sample_rate=args.rate)
                    label = f"effect:{effect}"
                check = check_signal(signal)
                out = args.output or os.path.join(
                    DEFAULT_OUT_DIR,
                    f"audio_{args.kind}_{_time.strftime('%Y%m%d_%H%M%S')}"
                    f"_{args.seed}.wav")
                out = resolve_output_path(out)
                info = write_wav(out, signal, sample_rate=args.rate)
                report = validate_wav(out)
            except Exception as exc:
                print(f"[creative] audio error: {exc}")
                return 1
            print(f"[creative] audio: {info['path']} ({label}, "
                  f"{info['sample_rate']}Hz, {info['duration']}s, "
                  f"{info['bytes'] // 1024}KB)")
            print(f"  signal: finite={check['finite']} "
                  f"peak={check['peak']} rms={check['rms']} "
                  f"silence={check['silence_ratio']} "
                  f"would_clip={check['would_clip']}")
            print(f"  verified: frames={report['frames']} "
                  f"clipping={report['clipping_count']} "
                  f"peak={report['peak']} duration={report['duration']}")
            return 0
        if args.creative_command == "benchmark":
            from media.backends import benchmark, BackendUnavailable
            try:
                report = benchmark(args.device, output_path=args.output)
            except BackendUnavailable as exc:
                print(f"[creative] benchmark: {exc}")
                return 1
            print(f"[creative] benchmark: device={report['device']} "
                  f"accelerated={report['accelerated']}")
            print(f"  {report['note']}")
            for name, timing in report["timings"].items():
                print(f"  {name}: {timing}")
            print(f"  report: {report.get('report_path', '')}")
            return 0
        if args.creative_command == "evaluate":
            from media.evaluate import evaluate_artifacts
            try:
                report = evaluate_artifacts(args.directory,
                                            output_path=args.output)
            except Exception as exc:
                print(f"[creative] evaluate error: {exc}")
                return 1
            print(f"[creative] evaluate: {report['directory']} "
                  f"artifacts={report['artifacts']} "
                  f"failed={report['failed']} "
                  f"images={report['by_kind']['images']} "
                  f"videos={report['by_kind']['videos']} "
                  f"audio={report['by_kind']['audio']} "
                  f"diversity={report['image_diversity']}")
            for record in report["records"][:12]:
                if "error" in record:
                    print(f"  ! {record['file']}: {record['error']}")
                elif record["file"].endswith(".png"):
                    print(f"  {record['file']}: {record['width']}x{record['height']} "
                          f"contrast={record['contrast']} entropy={record['entropy_bits']}")
                elif record["file"].endswith(".avi"):
                    print(f"  {record['file']}: {record['frames']}f {record['fps']}fps "
                          f"motion={record['motion_mean']}")
                else:
                    print(f"  {record['file']}: {record['duration']}s rms={record['rms']}")
            if report.get("report_path"):
                print(f"  report: {report['report_path']}")
            return 0
        if args.creative_command == "learn":
            from creative.learning import run_experiment
            prompts = [
                "a glass tower above an island",
                "a spiky machine beside a crystal",
                "three tiny stars on a rock",
                "a blue tower between two gates",
                "a wisp in a forest",
            ]
            try:
                metrics = run_experiment(prompts, output_path=args.output)
            except Exception as exc:
                print(f"[creative] learn error: {exc}")
                return 1
            print(f"[creative] learn: samples={metrics['samples']} "
                  f"learned={metrics['learned_accuracy']} "
                  f"majority={metrics['majority_baseline']} "
                  f"hand_rule={metrics['hand_rule_baseline']}")
            print(f"  train={metrics['train_prompts']}")
            print(f"  test={metrics['test_prompts']}")
            print(f"  report: {metrics.get('report_path', '')}")
            return 0
        if args.creative_command == "compose":
            from media.pipeline import compose
            from creative.planner import UnsupportedCreativeRequest
            try:
                meta = compose(
                    args.request, output_path=args.output,
                    duration=args.duration, fps=args.fps,
                    width=args.width, height=args.height,
                    samples=args.samples, seed=args.seed,
                    audio=args.audio)
            except UnsupportedCreativeRequest as exc:
                print(f"[creative] unsupported request: {exc}")
                return 1
            except Exception as exc:
                print(f"[creative] compose error: {exc}")
                return 1
            print(f"[creative] compose: {meta['path']} "
                  f"({meta['width']}x{meta['height']}, {meta['fps']}fps, "
                  f"{meta['frame_count']} frames, {meta['duration']}s, "
                  f"{meta['bytes'] // 1024}KB, "
                  f"{meta['compose_seconds']}s)")
            print(f"  audio: has_audio={meta['has_audio']} "
                  f"rate={meta['audio_sample_rate']} "
                  f"samples={meta['audio_samples']} "
                  f"peak={meta['audio_peak']} rms={meta['audio_rms']} "
                  f"clipping={meta['audio_clipping']}")
            print(f"  A/V sync error: {meta['sync_error_seconds']}s "
                  f"(tolerance <= max(1/fps, 0.05)) "
                  f"events={meta['audio_events']}")
            return 0
        if args.creative_command == "inspect-video":
            from media.video.analyze import analyze_video
            try:
                report = analyze_video(
                    args.input, output_path=args.output,
                    max_analysis_frames=args.max_frames,
                    change_threshold=args.threshold)
            except Exception as exc:
                print(f"[creative] inspect-video error: {exc}")
                return 1
            print(f"[creative] inspect-video: {report['format']} "
                  f"{report['width']}x{report['height']} "
                  f"fps={report['fps']} frames={report['frame_count']} "
                  f"duration={report['duration']}")
            print(f"  analyzed={report['analyzed_frames']} "
                  f"scene_changes={report['scene_change_count']} "
                  f"motion_mean={report['motion']['mean_diff']} "
                  f"motion_max={report['motion']['max_diff']}")
            if report.get("report_path"):
                print(f"  report: {report['report_path']}")
            return 0
        creative_parser.print_help()
    elif args.command == "computer":
        run_computer_command(args)


def run_computer_command(args) -> None:
    """Run the General Computer Agent on a natural-language task.

    The agent runs the full loop: UNDERSTAND -> OBSERVE -> REASON ->
    PLAN -> ACT -> OBSERVE -> VERIFY -> REPAIR -> LEARN, and verifies
    the observed screen state and task result — not merely that the
    commands were issued.
    """
    from computer.executor import ComputerAgent, create_backend
    from computer.security import ConfirmationPolicy

    confirm = None
    if getattr(args, "confirm", False):
        def confirm(action, description):
            print(f"\n[승인 필요] {description}")
            answer = input("실행할까요? [y/N]: ").strip().lower()
            return answer in ("y", "yes")

    agent = ComputerAgent(
        backend=create_backend(getattr(args, "backend", "auto"),
                               device_id=getattr(args, "device", None)),
        memory=None,
        policy=ConfirmationPolicy(confirm=confirm),
        progress=None if not getattr(args, "trace", False)
        else (lambda phase, detail:
              print(f"[{phase:>9}] {detail}")),
    )
    result = agent.execute_goal(args.request)

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    print(f"\n=== Computer Agent: {args.request} ===")
    goal = result.goal
    print(f"의도: {goal.intent.value}")
    if goal.application:
        print(f"대상 프로그램: {goal.application}")
    if result.trace:
        print("\n--- 실행 루프 ---")
        for t in result.trace:
            print(f"[{t['phase']:>9}] {t['detail']}")
    print("\n--- 액션 결과 ---")
    for r in result.action_results:
        at = r.action.action_type.value if r.action else "?"
        status = "OK" if r.success else "FAIL"
        print(f"  [{status}] {at}: {r.message[:100]}")
    if result.repairs:
        print(f"\n복구 시도: {len(result.repairs)}회")
    verdict = "달성" if result.success else "미달성"
    print(f"\n최종 검증: 목표 {verdict}")
    reasons = result.goal_verification.get("reasons", [])
    for reason in reasons[:3]:
        print(f"  - {reason}")
    if result.learnings:
        print("\n학습:")
        for l in result.learnings:
            print(f"  - [{l['memory_type']}] {l['content'][:100]}")


def run_e2e_test(brain: CodeBrain, verbose: bool = False):
    print("=" * 60)
    print("END-TO-END TEST: Natural Language -> IR -> Code -> Verify")
    print("=" * 60)

    test_cases = [
        {"request": "두 숫자를 더하는 함수 만들어줘",
         "languages": ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]},
        {"request": "팩토리얼 함수를 만들어줘",
         "languages": ["python", "javascript"]},
        {"request": "문자열을 결합하는 함수를 만들어줘",
         "languages": ["python", "go"]},
        {"request": "리스트에서 가장 큰 숫자를 찾는 함수를 만들어줘",
         "languages": ["python", "javascript", "kotlin"]},
        {"request": "피보나치 함수를 만들어줘",
         "languages": ["python"]},
        {"request": "소수 판별 함수를 만들어줘",
         "languages": ["python"]},
    ]

    total = 0
    passed = 0
    for tc in test_cases:
        print(f"\n--- Request: {tc['request']} ---")
        result = brain.process(tc["request"], tc["languages"][0])

        if verbose:
            print(f"Intent: {result['reasoning'].intent}")
            if result['reasoning'].structured_intent:
                si = result['reasoning'].structured_intent
                print(f"Algorithm: {si.algorithm}, Params: {si.params}, Return: {si.return_type}")
            print(f"Concepts: {[c.name for c in result['reasoning'].concepts]}")

        for name, data in result["results"].items():
            for lang in tc["languages"]:
                total += 1
                lang_data = data["languages"].get(lang, {})
                code = lang_data.get("code", "")
                has_code = bool(code) and "Error" not in code
                if has_code:
                    vresult = brain.verify(code, lang)
                    success = vresult.success
                else:
                    success = False
                    vresult = None

                status = "PASS" if success else "FAIL"
                if success:
                    passed += 1
                print(f"  [{lang.upper()}] {status}", end="")
                if not success and vresult and vresult.errors:
                    err = vresult.errors[0][:60]
                    if "not installed" in err.lower():
                        print(f" - toolchain not installed", end="")
                    else:
                        print(f" - {err}", end="")
                print()
                if verbose and has_code:
                    print(f"    Code: {code[:120]}...")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'=' * 60}")
    return passed == total


def print_streaming_result(result: dict):
    """Print the result of a streaming generation request with explanations."""
    # Show reasoning summary
    if "reasoning" in result:
        reasoning = result["reasoning"]
        if hasattr(reasoning, 'model'):
            model = reasoning.model
            goal_type = model.goal_type.value if model.goal_type else 'unknown'
            entities = [e.name for e in model.entities] if model.entities else []
            operations = [o.name for o in model.operations] if model.operations else []
            print(f"\n[분석] 목표: {goal_type}", end="")
            if entities:
                print(f" | 엔티티: {entities}", end="")
            if operations:
                print(f" | 연산: {operations}", end="")
            print()

            if reasoning.selected_hypothesis:
                hyp = reasoning.selected_hypothesis
                if hasattr(hyp, 'description'):
                    print(f"[접근법] {hyp.description} (신뢰도: {hyp.confidence:.2f})")
                elif isinstance(hyp, dict):
                    print(f"[접근법] {hyp.get('description', '')} (신뢰도: {hyp.get('confidence', 0):.2f})")

            # The Think trace: the budget's PUBLIC accounting — how many
            # candidates were considered, which alternative plans were
            # formulated, and whether the executed plan's provenance is
            # the selected hypothesis (the invariant).
            n_hyps = len(getattr(reasoning, "hypotheses", []) or [])
            alts = getattr(reasoning, "alternative_plans", []) or []
            src = getattr(reasoning, "plan_source_hypothesis", "")
            sel_id = getattr(reasoning.selected_hypothesis, "id", "")
            if n_hyps:
                print(f"[추론] 후보 {n_hyps}개 탐색"
                      + (f" | 대안 계획 {len(alts)}개 수립" if alts else "")
                      + (" | 계획 근원=선택 가설 ✓" if src and src == sel_id
                         else ""))
            for a in alts[:2]:
                print(f"        대안: {a.get('hypothesis')} "
                      f"({len(a.get('plan', []))} steps)")

            if hasattr(reasoning, 'steps') and reasoning.steps:
                for ev in reasoning.steps:
                    stage = getattr(ev, 'stage', '')
                    msg = getattr(ev, 'message', '')
                    if msg:
                        print(f"  {msg}")

    # Show streaming generation events
    if "generation_events" in result and result["generation_events"]:
        print(f"\n--- 실시간 생성 과정 ---")
        current_file = None
        for event in result["generation_events"]:
            event_type = event.get("event_type", "")
            file_path = event.get("file_path", "")
            line_number = event.get("line_number", 0)
            code = event.get("code", "").rstrip()
            explanation = event.get("explanation", {})
            
            if event_type == "start":
                print(f"\n🚀 코드 생성 시작")
                if explanation:
                    print(f"   💭 {explanation.get('korean_explanation', '')}")
            
            elif event_type == "file_start":
                if file_path != current_file:
                    current_file = file_path
                    print(f"\n📄 {file_path}")
                if explanation:
                    print(f"   💭 {explanation.get('korean_explanation', '')}")
            
            elif event_type == "class_start":
                name = event.get("structure_context", {}).get("current_class", "")
                print(f"  ┌ 클래스: {name}")
                if explanation:
                    print(f"  │ 💭 {explanation.get('korean_explanation', '')}")
            
            elif event_type == "function_start":
                name = event.get("structure_context", {}).get("current_function", "")
                print(f"  ├─ 함수: {name}")
                if explanation:
                    print(f"  │ 💭 {explanation.get('korean_explanation', '')}")
            
            elif event_type == "line":
                if code:
                    print(f"  │ {line_number:4d} │ {code}")
                    if explanation and explanation.get('korean_explanation'):
                        expl = explanation['korean_explanation']
                        if len(expl) < 80:
                            print(f"  │      💭 {expl}")
            
            elif event_type == "structure_end":
                print(f"  └ 완료")
            
            elif event_type == "complete":
                print(f"\n✅ 생성 완료")
                if explanation:
                    print(f"   💭 {explanation.get('korean_explanation', '')}")

    # Show verification result
    if "verification" in result:
        v = result["verification"]
        status = "PASS" if v["success"] else "FAIL"
        print(f"\n[검증] {status} (통과: {v['checks_passed']}, 실패: {v['checks_failed']})")

    # Show generated project files
    if result.get("project_created") and result.get("files"):
        print(f"\n--- 프로젝트 생성: {result.get('request', '')} ---")
        for f in result["files"][:5]:
            print(f"\n  [{f['file_type']}] {f['path']}")
            for line in f.get("content", "").split("\n")[:10]:
                print(f"    {line}")


def _print_think_header(budget) -> None:
    """The request's reasoning budget, stated up front — the PUBLIC
    parameters of the reasoning (candidates, plans, verification
    inputs, repair attempts), never hidden chain-of-thought."""
    if budget.mode.value == "normal":
        return
    label = "Think" if budget.mode.value == "think" else "Deep Think"
    print(f"[{label}] 추론 예산: 후보 {budget.max_hypotheses} | "
          f"대안 계획 {budget.max_alternative_plans} | "
          f"검증 입력 {budget.max_verification_tests} | "
          f"수리 시도 {budget.max_repair_attempts}")


def _ordered_languages(langs: dict, declared: str = "") -> list:
    """The declared target first; every other language keeps its own
    generation order.  The listing follows the REQUEST's declared
    target, never the renderer's dictionary order."""
    keys = list(langs.keys())
    if declared and declared in keys:
        return [declared] + [k for k in keys if k != declared]
    return keys


def _declared_language(request: str) -> str:
    try:
        from brain.target_detector import detect_target
        return detect_target(request or "").language or ""
    except Exception:
        return ""


def print_natural_result(result: dict):
    """Print the result of a natural language request."""
    # Audio artifacts (natural-language music/effects) report their
    # measured signal properties.
    if isinstance(result.get("audio"), dict):
        audio = result["audio"]
        print(f"\n[오디오] {audio.get('path')} "
              f"({audio.get('label')}, {audio.get('sample_rate')}Hz, "
              f"{audio.get('duration')}s, {audio.get('bytes', 0) // 1024}KB)")
        print(f"  검증: peak={audio.get('peak')} "
              f"rms={audio.get('rms')} "
              f"clipping={audio.get('clipping')} "
              f"silence={audio.get('silence_ratio')} "
              f"finite={audio.get('finite')}")
        if result.get("response"):
            print(f"\n{result['response']}")
        _conf = result.get("confidence")
        if isinstance(_conf, dict) and "confidence" in _conf:
            print(f"\n[신뢰도] {_conf.get('confidence')} — "
                  f"{_conf.get('basis')}")
        return
    # Image artifacts (natural-language drawing) carry their own report.
    if isinstance(result.get("image"), dict):
        image = result["image"]
        print(f"\n[이미지] {image.get('path')} "
              f"({image.get('width')}x{image.get('height')})")
        if image.get("summary"):
            print(f"  {image['summary']}")
        if result.get("response"):
            print(f"\n{result['response']}")
        _conf = result.get("confidence")
        if isinstance(_conf, dict) and "confidence" in _conf:
            print(f"\n[신뢰도] {_conf.get('confidence')} — "
                  f"{_conf.get('basis')}")
        return
    declared = _declared_language(result.get("request", "") or "")
    # Show generated code first when the algorithmic pipeline produced it
    # (its result dict carries "reasoning": None by design)
    if result.get("algorithmic") and "results" in result and result["results"]:
        for name, data in result["results"].items():
            for lang in _ordered_languages(data.get("languages", {}), declared):
                lang_data = data.get("languages", {}).get(lang, {})
                code = lang_data.get("code", "")
                if code:
                    print(f"\n--- 생성된 코드 ---")
                    print(f"\n# {lang.upper()}")
                    print(code)
                    vr = result.get("verification")
                    if vr is not None:
                        print(f"  [검증: {'PASS' if vr else 'FAIL'}]")
            break
        if result.get("response"):
            print(f"\n{result['response']}")
        return
    # Show reasoning summary
    if "reasoning" in result:
        reasoning = result["reasoning"]
        if reasoning is None:
            pass
        elif hasattr(reasoning, 'model'):
            model = reasoning.model
            goal_type = model.goal_type.value if model.goal_type else 'unknown'
            entities = [e.name for e in model.entities] if model.entities else []
            operations = [o.name for o in model.operations] if model.operations else []
            print(f"\n[분석] 목표: {goal_type}", end="")
            if entities:
                print(f" | 엔티티: {entities}", end="")
            if operations:
                print(f" | 연산: {operations}", end="")
            print()

            if reasoning.selected_hypothesis:
                hyp = reasoning.selected_hypothesis
                if hasattr(hyp, 'description'):
                    print(f"[접근법] {hyp.description} (신뢰도: {hyp.confidence:.2f})")
                elif isinstance(hyp, dict):
                    print(f"[접근법] {hyp.get('description', '')} (신뢰도: {hyp.get('confidence', 0):.2f})")

            # The Think trace: the budget's PUBLIC accounting — how many
            # candidates were considered, which alternative plans were
            # formulated, and whether the executed plan's provenance is
            # the selected hypothesis (the invariant).
            n_hyps = len(getattr(reasoning, "hypotheses", []) or [])
            alts = getattr(reasoning, "alternative_plans", []) or []
            src = getattr(reasoning, "plan_source_hypothesis", "")
            sel_id = getattr(reasoning.selected_hypothesis, "id", "")
            if n_hyps:
                print(f"[추론] 후보 {n_hyps}개 탐색"
                      + (f" | 대안 계획 {len(alts)}개 수립" if alts else "")
                      + (" | 계획 근원=선택 가설" if src and src == sel_id
                         else ""))
            for a in alts[:2]:
                print(f"        대안: {a.get('hypothesis')} "
                      f"({len(a.get('plan', []))} steps)")

            # Show reasoning events if available
            if hasattr(reasoning, 'steps') and reasoning.steps:
                for ev in reasoning.steps:
                    stage = getattr(ev, 'stage', '')
                    msg = getattr(ev, 'message', '')
                    if msg:
                        print(f"  {msg}")
        else:
            # Old dict format
            model = reasoning.get("model", {})
            print(f"\n[분석] 목표: {model.get('goal_type', 'unknown')}", end="")
            if model.get("entities"):
                print(f" | 엔티티: {[e['name'] for e in model['entities']]}", end="")
            if model.get("operations"):
                print(f" | 연산: {[o['name'] for o in model['operations']]}", end="")
            print()

            if "selected_hypothesis" in reasoning and reasoning["selected_hypothesis"]:
                hyp = reasoning["selected_hypothesis"]
                print(f"[접근법] {hyp.get('description', '')} (신뢰도: {hyp.get('confidence', 0):.2f})")

            if "reasoning_events" in result and result["reasoning_events"]:
                for ev in result["reasoning_events"]:
                    stage = ev.get("stage", "")
                    msg = ev.get("message", "")
                    if msg:
                        print(f"  {msg}")

    # Show errors
    if result.get("intent") == "ERROR":
        print(f"\n[오류] {result.get('response', '')}")
        return

    # Show generated code
    if "results" in result and result["results"]:
        print(f"\n--- 생성된 코드 ---")
        for name, data in result["results"].items():
            for lang in _ordered_languages(data.get("languages", {}), declared):
                lang_data = data.get("languages", {}).get(lang, {})
                code = lang_data.get("code", "")
                if code and "Error" not in code:
                    print(f"\n# {lang.upper()}")
                    print(code)
                    vr = lang_data.get("verification")
                    if vr:
                        status = "PASS" if vr.get("success") else "FAIL"
                        detail = " (행동 검증 포함)" if vr.get("behavioral") else ""
                        if vr.get("repaired"):
                            detail += f" (수리 후 통과, {vr.get('repair_attempts', 1)}회 시도)"
                        print(f"  [검증: {status}{detail}]")
            break  # Only show first language

    # The FINAL confidence: what verification actually established —
    # never the raw semantic plausibility presented as the result.
    _conf = result.get("confidence")
    if isinstance(_conf, dict) and "confidence" in _conf:
        print(f"\n[신뢰도] {_conf.get('confidence')} — {_conf.get('basis')}")

    # Requirement Coverage Matrix (the final PASS authority)
    _cov = result.get("requirement_coverage")
    if isinstance(_cov, dict) and _cov.get("items"):
        try:
            if "compiles" in _cov:
                from brain.project_requirements import format_project_coverage
                print()
                print(format_project_coverage(_cov))
            else:
                from brain.requirement_contract import format_coverage
                print()
                print(format_coverage(_cov))
        except Exception:
            pass

    # Show generated project
    if result.get("project_created") and result.get("files"):
        print(f"\n--- 프로젝트 생성: {result.get('request', '')} ---")
        if "verification" in result:
            v = result["verification"]
            status = "PASS" if v["success"] else "FAIL"
            print(f"  상태: {status} (통과: {v['checks_passed']}, 실패: {v['checks_failed']})")
        for f in result["files"][:5]:
            print(f"\n  [{f['file_type']}] {f['path']}")
            for line in f.get("content", "").split("\n")[:10]:
                print(f"    {line}")

    if "response" in result and result["response"]:
        chunks = result.get("response_chunks") or [result["response"]]
        print()
        interactive = sys.stdout.isatty()
        # a chunk is a sentence, and the chunker trims the whitespace
        # that held it to its neighbours: the display puts that
        # separator back so the printed response reads as written
        shown = ""
        for chunk in chunks:
            piece = chunk
            if shown and not shown[-1].isspace() and not piece[:1].isspace():
                piece = " " + piece
            shown += piece
            print(piece, end="", flush=True)
            if interactive:
                time.sleep(0.15)
        print()


def print_debug_result(result: dict):
    """Print the result with full debug information."""
    debug = result.get("debug", {})

    print("=" * 60)
    print("DEBUG: Internal State")
    print("=" * 60)

    for section, data in debug.items():
        print(f"\n{section}")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 5:
                    print(f"  {k}: [{', '.join(str(x) for x in v[:5])}... ({len(v)} total)]")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for dk, dv in v.items():
                        print(f"    {dk}: {dv}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    desc = item.get("description", item.get("name", str(item)))
                    conf = item.get("confidence", "")
                    if conf:
                        print(f"  - {desc} (conf={conf:.2f})")
                    else:
                        print(f"  - {desc}")
                else:
                    print(f"  - {item}")
        else:
            print(f"  {data}")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    # Print normal result
    print_natural_result(result)


def _trace_history(brain, limit: int = 4) -> list:
    """The conversation's own record: the last turns as the pipeline
    stored them — the structured meaning the next resolution reads."""
    try:
        turns = list(brain.conversation_manager.history)
    except Exception:
        return []
    out = []
    for turn in turns[-limit:]:
        knowledge = getattr(turn, "referenced_knowledge", None) or {}
        out.append({
            "user_input": getattr(turn, "user_input", ""),
            "intent": getattr(turn, "intent", ""),
            "topic": getattr(turn, "topic", ""),
            "topics": list(getattr(turn, "topics", []) or []),
            "entities": list(getattr(turn, "entities", []) or [])[:6],
            "claims": list(getattr(turn, "claims", []) or [])[:2],
            "referenced_knowledge": {
                k: knowledge.get(k) for k in ("target", "query")
                if knowledge.get(k)},
        })
    return out


def _reference_threshold() -> float:
    """The confidence below which a reference asks instead of guessing."""
    try:
        from brain.context_resolution import CLARIFICATION_THRESHOLD
        return round(float(CLARIFICATION_THRESHOLD), 3)
    except Exception:
        return 0.45


def trace_algorithmic(brain: CodeBrain, request: str):
    """Trace a request through the understanding → route → pipeline.

    The stages are printed in the order the pipeline runs them: the
    semantic understanding, the INTENT (the act), the CONTEXT/REFERENCE
    resolution and the TOPIC STATE — those decide the ROUTE.  Only a
    CODE route runs the coding pipeline (hypotheses → reasoning → plan
    → IR → code); every other route runs the knowledge / conversation
    pipeline (context → knowledge → response) and never the algorithmic
    one.
    """
    import time
    import json as json_mod

    def _dump(title: str, data):
        print(title)
        text = json_mod.dumps(data, indent=2, ensure_ascii=False,
                              default=str)
        for line in text.split("\n"):
            print(f"    {line}")
        print()

    def _dump_code_artifacts(result: dict):
        results = result.get("results") or {}
        emitted = False
        for name, payload in results.items():
            if not isinstance(payload, dict) or "ir" not in payload:
                continue
            emitted = True
            _dump(f"[10] IR ({name})", payload.get("ir") or
                  {"status": "no IR"})
            languages = payload.get("languages") or {}
            if languages:
                primary = next(iter(languages))
                entry = languages[primary] or {}
                _dump("[11] CODE", {
                    "language": primary,
                    "code": entry.get("code") if isinstance(entry, dict)
                            else entry,
                })
            _dump("[12] VERIFICATION",
                  payload.get("verification") or
                  {"status": "no verification"})
        if not emitted:
            _dump("[10] IR", {"status": "no IR in the result"})

    t0 = time.time()
    result = brain.process_semantic(request)
    t_run = time.time() - t0

    trace = result.get("trace") or {}
    semantic = result.get("semantic") or {}
    intent = trace.get("intent") or {}
    reasoning = trace.get("reasoning") or {}
    route = trace.get("route") or result.get("route") or ""
    from brain.route import describe_route

    print("=" * 60)
    print("RENA-BR TRACE")
    print("=" * 60)
    print()

    _dump("[01] RAW INPUT", {"request": request})

    import re as _re

    goal = trace.get("goal") or {}
    entities = [{"name": e.get("name"),
                 "type": e.get("type") or e.get("type_name")}
                for e in (semantic.get("entities") or [])
                if isinstance(e, dict)]
    operations = [{"name": o.get("name"), "type": o.get("type"),
                   "role": o.get("semantic_role"),
                   "params": o.get("params") or {}}
                  for o in ((semantic.get("operation_graph") or {})
                            .get("operations") or [])]

    # The RELATION CHAIN: every link below is a declared part of the
    # semantics — input type, the element type the collection declares,
    # the measure applied to that element, the objective over the
    # measure and the output type.  A link that was never declared is
    # omitted instead of invented, so the chain is a faithful view of
    # what was understood (and of what was NOT).
    inputs = [i for i in (semantic.get("inputs") or []) if isinstance(i, dict)]
    outputs = [o for o in (semantic.get("outputs") or []) if isinstance(o, dict)]
    objectives = [o for o in (semantic.get("objectives") or [])
                  if isinstance(o, dict) and (o.get("direction")
                                              or o.get("measure"))]
    objective = objectives[0] if objectives else {}
    element = ""
    in_types = [str(i.get("type") or "") for i in inputs]
    for t in in_types:
        m = _re.match(r"(?i)^\s*list\s*<\s*(.+?)\s*>\s*$", t) or \
            _re.match(r"(?i)^\s*list\s*\[\s*(.+?)\s*\]\s*$", t)
        if m:
            element = m.group(1).strip()
            break
    measure = str(objective.get("measure") or "")
    direction = str(objective.get("direction") or "")
    chain = []
    if in_types:
        chain.append("input: " + ", ".join(in_types))
    if element:
        chain.append(f"element: {element}")
    if measure:
        subject = element or "element"
        chain.append(f"{measure}({subject})")
    if direction:
        chain.append("objective: "
                     + (f"{direction}({measure})" if measure else direction))
    signature = semantic.get("function_signature") or {}
    out_type = str(semantic.get("output_type") or "")
    if not out_type:
        out_types = [str(o.get("type") or "") for o in outputs]
        out_type = ", ".join(t for t in out_types if t) or             str(signature.get("return_type") or "")
    if out_type:
        chain.append(f"output: {out_type}")

    _dump("[02] SEMANTIC UNDERSTANDING", {
        "goal_type": goal.get("type"),
        "goal_target": goal.get("target"),
        "goal_confidence": goal.get("confidence"),
        "parse_confidence": semantic.get("confidence"),
        "objective": objective or None,
        "relation_chain": chain,
        "io": {
            "inputs": [{"name": i.get("name"), "type": i.get("type")}
                       for i in inputs],
            "outputs": [{"name": o.get("name"), "type": o.get("type")}
                        for o in outputs],
            "output_type": out_type,
        },
        "function_signature": {
            "name": signature.get("name"),
            "parameters": [{"name": p.get("name"),
                            "type": p.get("type") or p.get("type_name")}
                           for p in (signature.get("parameters") or [])],
            "return_type": signature.get("return_type"),
        } if signature else None,
        "entities": entities,
        "operations": operations,
    })

    # The understanding is COMPLETE: the goal, the act's intent and the
    # content topics are all decided — this is the point past which no
    # stage may re-derive them, only read them.
    _dump("[02b] SEMANTIC COMPLETION", {
        "status": "complete",
        "goal_type": goal.get("type"),
        "intent": intent.get("type"),
        "intent_goal": intent.get("goal"),
        "topics": intent.get("topics"),
        "parse_confidence": semantic.get("confidence"),
        "route": route,
    })

    _dump("[03] INTENT", {
        "type": intent.get("type"),
        "goal": intent.get("goal"),
        "confidence": intent.get("confidence"),
        "evidence": intent.get("evidence"),
        "act": intent.get("act"),
        "mapped_intent": intent.get("legacy_intent"),
    })

    _dump("[04] CONTEXT / REFERENCE", trace.get("context") or {})
    _dump("[05] TOPIC STATE", trace.get("topic") or {})
    _dump("[06] ROUTE", {"route": route,
                         "pipeline": describe_route(
                             route, str(intent.get("type") or ""))})
    # The conversation's own stages, in the order the pipeline ran
    # them: the history it read, the topic state, the reference
    # candidates, the reference it resolved with its confidence, the
    # discourse operation, the knowledge requirement, the retrieval,
    # and the response representation.  Every stage is the SAME
    # structure the pipeline acted on — never a re-derivation.
    reference = trace.get("reference_resolution") or {}
    requirement = trace.get("knowledge_requirement") or {}
    web_search = trace.get("web_search") or {}
    retrieval = trace.get("retrieved_knowledge") or {}
    _dump("[06a] CONVERSATION HISTORY", _trace_history(brain))
    _dump("[06b] TOPIC STATE", trace.get("topic_state") or {})
    _dump("[06c] REFERENCE CANDIDATES", [{
        "target": c.get("target"),
        "kind": c.get("kind"),
        "source": c.get("source"),
        "score": c.get("score"),
        "reasons": c.get("reasons"),
    } for c in (reference.get("candidates") or [])])
    _dump("[06d] RESOLVED REFERENCE", {
        "surface_form": reference.get("surface_form"),
        "refers_to": reference.get("refers_to"),
        "target": reference.get("target"),
        "target_kind": reference.get("target_kind"),
        "needs_clarification": reference.get("needs_clarification"),
        "evidence": reference.get("evidence"),
    })
    _dump("[06e] REFERENCE CONFIDENCE", {
        "confidence": reference.get("confidence"),
        "threshold": _reference_threshold(),
        "resolved": bool(reference.get("target")) and
                    not bool(reference.get("needs_clarification")),
        "candidates": len(reference.get("candidates") or []),
    })
    _dump("[06f] DISCOURSE ACTION",
          trace.get("discourse_action") or {})
    _dump("[06g] KNOWLEDGE REQUIREMENT", {
        "need_web": requirement.get("need_web"),
        "need_local": requirement.get("need_local"),
        "need_context": requirement.get("need_context"),
        "need_self": requirement.get("need_self"),
        "reason": requirement.get("reason"),
        "signals": requirement.get("signals"),
        "freshness": requirement.get("freshness"),
        "time_scope": requirement.get("time_scope"),
        "query": requirement.get("query"),
    })
    _dump("[06h] WEB SEARCH", {
        "used": web_search.get("used"),
        "reason": web_search.get("reason"),
        "query": web_search.get("query"),
        "results": web_search.get("results"),
    })
    _dump("[06i] RETRIEVED KNOWLEDGE", {
        "query": retrieval.get("query"),
        "web_used": retrieval.get("web_used"),
        "reason": retrieval.get("reason"),
        # The research loop's record: every query it asked (including
        # the reformulated ones) and how many attempts it took.
        "queries": retrieval.get("queries"),
        "attempts": retrieval.get("attempts"),
        "gated": retrieval.get("gated"),
        "evidence": [{
            "source": e.get("source"),
            "title": e.get("title"),
            "relevance": e.get("relevance"),
            "confidence": e.get("confidence"),
            "freshness": e.get("freshness"),
            "verification_status": e.get("verification_status"),
        } for e in (retrieval.get("evidence") or [])],
    })
    _dump("[06j] RESPONSE REPRESENTATION",
          trace.get("response_representation") or {})
    # The two stages after the representation: WHAT the response will
    # say (the sentence plan over the representation's moves) and WHAT
    # it ended up saying (the realization) — kept apart so the plan
    # and the wording can be compared, never conflated.
    representation = trace.get("response_representation") or {}
    knowledge = trace.get("knowledge") or {}
    _dump("[06k] RESPONSE COMPOSITION", {
        "claims": knowledge.get("claims"),
        "moves": representation.get("moves"),
        "sentence_plan": representation.get("sentence_plan"),
    })
    _dump("[06l] REALIZATION", {
        "language": representation.get("language"),
        "style": representation.get("style"),
        "text": result.get("response"),
    })
    # The output generation: the length budget this response obeyed,
    # and the long-form report (sections planned, verification,
    # revision) — empty on the short path.
    _dump("[06m] LONGFORM GENERATION",
          trace.get("generation") or {})
    print()

    if route == "CODE":
        _dump("[07] HYPOTHESES", [{
            "id": h.get("id"),
            "description": h.get("description"),
            "confidence": h.get("confidence"),
        } for h in (reasoning.get("hypotheses") or [])])
        _dump("[08] REASONING", {
            "selected": reasoning.get("selected"),
            "confidence": reasoning.get("confidence"),
            "evaluation": reasoning.get("evaluation"),
        })
        _dump("[09] PLAN", reasoning.get("plan") or [])
        _dump_code_artifacts(result)
        stage = "[13] RESPONSE"
    else:
        knowledge = trace.get("knowledge") or {}
        _dump("[07] HYPOTHESES", {
            "generated": False,
            "route": route,
            "reason": "non-coding route: the turn carries no problem "
                      "state, so the Code Hypothesis Generator is "
                      "never called",
        })
        _dump("[08] REASONING (KNOWLEDGE)", {
            "basis": "knowledge",
            "target": knowledge.get("target"),
            "evidence": knowledge.get("evidence"),
            "claims": knowledge.get("claims"),
        })
        stage = "[09] RESPONSE"

    _dump(stage, {
        "text": result.get("response"),
        "chunks": result.get("response_chunks"),
    })

    # The TURN as the conversation recorded it: what was asked, how it
    # was understood, what it was about and what it said — the record
    # the NEXT turn resolves against.
    topic = trace.get("topic") or {}
    _dump("[14] TURN", {
        "input": trace.get("input"),
        "intent": intent.get("type"),
        "topics": intent.get("topics"),
        "claims": knowledge.get("claims"),
        "response": result.get("response"),
    })
    # The STATE UPDATE: the topic state before and after this turn —
    # a turn that resolved a referent moves the discourse, a
    # clarification never does.
    _dump("[15] STATE UPDATE", {
        "topic_before": topic.get("before"),
        "topic_after": topic.get("after"),
        "state": trace.get("topic_state") or {},
        "discourse_action": trace.get("discourse_action") or {},
    })

    print("=" * 60)
    print(f"PIPELINE ELAPSED: {t_run:.3f}s")
    print("=" * 60)


def run_voice_conversation():
    """Real-time voice conversation mode: listen -> STT -> RENA-BR -> TTS.

    The speech backends are probed honestly first; a missing capability
    is reported, never assumed.
    """
    from computer.agents.voice_agent import VoiceAgent
    from brain.code_brain import CodeBrain

    brain = CodeBrain()
    agent = VoiceAgent(
        process_fn=lambda request: brain.process_semantic(request, "en")
                                  .get("response", ""),
    )

    # the banner goes through the VoiceOutput layer too: console output
    # is best-effort and never takes the Voice Agent down
    out = agent.output
    out.write("=" * 60)
    out.write("RENA-BR VOICE CONVERSATION")
    out.write("=" * 60)
    info = agent.backend_info
    out.write(f"  STT: {info.stt_backend or 'unsupported'} "
              f"({'ready' if info.stt_ready else 'NOT ready'})")
    out.write(f"  TTS: {info.tts_backend or 'unsupported'} "
              f"({'ready' if info.tts_ready else 'NOT ready'})")
    for k, v in info.details.items():
        out.write(f"    {k}: {v}")
    if not info.stt_ready:
        out.write("\n음성 입력 백엔드가 이 환경에 없습니다.")
        out.write("설치: /usr/bin/python3 -m pip install vosk (오프라인 STT)")
        out.write("      apt-get install espeak-ng (TTS)")
        return
    if not info.tts_ready:
        out.write("\n[주의] 음성 출력 백엔드가 없습니다 — 텍스트로만 응답합니다.")
    agent.run_conversation()


if __name__ == "__main__":
    main()
