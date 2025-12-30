# src/application/kimina_client_pool.py
from __future__ import annotations
from typing import List, Dict
from kimina_client import KiminaClient                # kimina HTTP client
from kimina_client.models import Snippet
import os
import re
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
import threading
from collections import deque

class KiminaPool:
    """
    Wrap a single Lean4Client and expose three batched helpers:
        • compile_only      – Lean kernel, no tactics
        • exact_suggestion  – append 'exact?' and compile
        • aesop_suggestion  – append 'aesop?' and compile
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
        num_proc: int = 16,
        batch_size: int = 32,
    ):
        """Thin wrapper around :class:`Lean4Client` with sensible defaults.

        Parameters
        ----------
        base_url : str | None
            Base URL of the kimina-lean-server, including scheme & port.  If
            *None*, the environment variable ``KIMINA_SERVER_URL`` is honoured
            (if present); otherwise we default to ``http://localhost:12332``
            which is the port used by the bundled Lean server implementation.
        timeout : int
            Request timeout forwarded to the underlying client (seconds).
        num_proc : int
            *Unused for now* – placeholder for future pool parallelism.
        batch_size : int
            Batch size forwarded to the map-elites evaluator (not yet used).
        """

        if base_url is None:
            base_url = os.getenv("KIMINA_SERVER_URL", "http://localhost")
        print(f"Using Kimina server at {base_url}")
        self._client  = KiminaClient(api_url=base_url)
        self._timeout = timeout
        self._num_proc = num_proc
        self._batch_size = batch_size

    # ------------------- NEW analysis helper -------------------
    def analyze_statement(self, code: str):
        """Forward a single Lean statement to the /analyze endpoint.

        This is used by src.application.descriptors to obtain parser-aware
        features for MAP-Elites.
        """
        return self._client.analyze_statement(code, timeout=self._timeout)

    # ------------ low‑level RPC helpers -----------------
    def _verify_batch(self, items: List[Dict]) -> List[Dict]:
        """Return the list of results from a Kimina `/check` call.

        Kimina-client >=0.6.0 returns a `CheckResponse` object with a
        `.results` attribute instead of the old `dict` payload.  To stay
        backward-compatible we support both shapes.
        """
        resp = self._client.check(
            items,
            timeout=self._timeout,
            batch_size=self._batch_size,
            max_workers=self._num_proc,
            debug=True,
            show_progress=False,
        )
        # Newer kimina_client returns a dataclass-like object
        from kimina_client.models import backward_response_from_repl
        if hasattr(resp, "results"):
            # Convert each `ReplResponse` to the old backward-compatible dict shape
            converted = []
            for r in resp.results:
                d = backward_response_from_repl(r)
                # Ensure keys exist so older code using "['error']" doesn't crash
                if "error" not in d:
                    d["error"] = None
                if "response" not in d or d["response"] is None:
                    d["response"] = {"messages": []}
                converted.append(d)
            return converted  # type: ignore[attr-defined]
        # Older versions returned a plain dict list already
        return resp["results"]

    def _inject(self, wrapper: str, i: int, body: str, j: int | None = None) -> str:
        ns = f"__kimina_tmp_{i}" if j is None else f"__kimina_tmp_{i}_{j}"
        return wrapper.replace("[CONJECTURE]", f"namespace {ns}\n{body}\nend {ns}")
    
    # ------------ public helpers -----------------------
    def compile_only(self, snippets, wrapper):
        req = [Snippet(id=str(i),
                       code=self._inject(wrapper, i, s.rstrip() + " sorry"))
               for i, s in snippets.items()]
        return self._verify_batch(req)
    
    def compile(self, snippets, wrapper):
        req = [Snippet(id=str(i),
                       code=self._inject(wrapper, i, s.rstrip()))
               for i, s in snippets.items()]
        return self._verify_batch(req)

    def exact_suggestion(self, snippets, wrapper):
        req = [Snippet(id=str(i),
                       code=self._inject(wrapper, i, s.rstrip() + " exact?\n"))
               for i, s in snippets.items()]
        return self._verify_batch(req)

    def aesop_suggestion(self, snippets, wrapper):
        req = [Snippet(id=str(i),
                       code=self._inject(wrapper, i, s.rstrip() + " aesop?\n"))
               for i, s in snippets.items()]
        return self._verify_batch(req)

    def verify_proofs_passk(
        self,
        conjectures: Dict[int, str],
        proofs_per_conj: Dict[int, List[str]],
        wrapper: str,
    ) -> List[List[Dict]]:
        """
        For each statement[i] and each proof tail proofs_per_conj[i][j], we build:

            code = f"{statement}\n{proof_tail}\n"

        We send ONE batched Kimina /check call with Snippet ids "{i}:{j}".
        We return results grouped & aligned as a list[ list[ dict ] ] so that
        results[i][j] corresponds to proofs_per_conj[i][j].
        """
        assert set(conjectures.keys()) == set(proofs_per_conj.keys()), \
        "conjectures and proofs_per_conj must have same keys"

        items: List[Snippet] = []
        for i in conjectures.keys():
            stmt = conjectures[i].rstrip()
            for j, prf in enumerate(proofs_per_conj[i]):
                body = f"{stmt}\n{prf.rstrip()}\n"
                # print("stmt: ", stmt)
                # print("prf: ", prf)
        
                # print("body: ", body)
                
                code = self._inject(wrapper, i, body, j=j)   # <— use wrapper + unique ns
                # print("wrapped code: ", code)
                # print("--------------------------------")
                items.append(Snippet(id=f"{i}:{j}", code=code))

        flat_results = self._verify_batch(items)

        grouped: Dict[int, List[Dict]] = {}
        id_re = re.compile(r"^(-?\d+):(\d+)$")
        for r in flat_results:
            cid = r.get("custom_id")
            m = id_re.match(str(cid))
            if not m:
                raise RuntimeError(f"Unexpected custom_id shape: {cid}")
            i, j = int(m.group(1)), int(m.group(2))
            lst = grouped.setdefault(i, [])
            while len(lst) <= j:
                lst.append({})
            lst[j] = r
        return grouped
    
    def verify_proofs_passk_batch(
        self,
        items: List[Dict],
    ) -> List[List[Dict]]:
        """
        For each item[i], we build:

            code = f"{item['code']}\n"

        We send ONE batched Kimina /check call with Snippet ids "{i}:{j}".
        We return results grouped & aligned as a list[ list[ dict ] ] so that
        results[i] corresponds to items[i].
        """
     

        snippet_items: List[Snippet] = []
        for item in items:
            snippet_items.append(Snippet(id=f"{item['id']}", code=item['code']))

        flat_results = self._verify_batch(snippet_items)

        grouped: Dict[int, List[Dict]] = {}
        id_re = re.compile(r"^(-?\d+):(\d+)$")
        for r in flat_results:
            cid = r.get("custom_id")
            m = id_re.match(str(cid))
            if not m:
                raise RuntimeError(f"Unexpected custom_id shape: {cid}")
            i, j = int(m.group(1)), int(m.group(2))
            lst = grouped.setdefault(i, [])
            while len(lst) <= j:
                lst.append({})
            lst[j] = r
        return grouped

    def verify_proofs_passk_queued(
        self,
        items: List[Dict],
        num_slots: int = 25,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> List[List[Dict]]:
        """
        Send snippets one-at-a-time, maintaining per-section queues.
        At any time, up to `num_slots` sections are active (one per slot).
        When a section queue empties, the slot is reassigned to another section.
        If all sections are already active (no new headers remain to assign),
        idle workers will steal work from the section with the most items left.

        items: list of {id: "i:j", code: str, header: str}
        Returns results grouped by i -> list[j] of response dicts (backward-compatible shape).
        """
        # Group items by header (section wrapper)
        by_header: Dict[str, deque[Snippet]] = {}
        for it in items:
            hdr = it.get("header", "")
            if hdr not in by_header:
                by_header[hdr] = deque()
            by_header[hdr].append(Snippet(id=str(it["id"]), code=it["code"]))

        # Prepare coordinator state
        headers = [h for h, q in by_header.items() if len(q) > 0]
        header_queue: deque[str] = deque(headers)
        assigned: set[str] = set()
        state_lock = threading.Lock()

        grouped: Dict[int, List[Dict]] = {}
        id_re = re.compile(r"^(-?\d+):(\d+)$")

        def acquire_header() -> str | None:
            with state_lock:
                while header_queue:
                    h = header_queue.popleft()
                    if h in assigned:
                        # Should not happen, but skip defensively
                        continue
                    if by_header[h]:
                        assigned.add(h)
                        return h
                return None

        def acquire_heaviest_header() -> str | None:
            """Pick the header with the most remaining items (work stealing).

            This allows more than one worker to drain the same header when all
            distinct headers have already been assigned once. We intentionally
            do not mutate `assigned` here so that the originally pinned worker
            remains considered assigned for that header.
            """
            with state_lock:
                # Build list of (header, remaining_count) for non-empty queues
                candidates = [(h, len(q)) for h, q in by_header.items() if len(q) > 0]
                if not candidates:
                    return None
                # Choose the header with the most items remaining
                candidates.sort(key=lambda t: t[1], reverse=True)
                return candidates[0][0]

        def release_header(h: str) -> None:
            with state_lock:
                assigned.discard(h)

        def record_result(r: Dict) -> None:
            cid = r.get("custom_id")
            m = id_re.match(str(cid))
            if not m:
                raise RuntimeError(f"Unexpected custom_id shape: {cid}")
            i, j = int(m.group(1)), int(m.group(2))
            with state_lock:
                lst = grouped.setdefault(i, [])
                while len(lst) <= j:
                    lst.append({})
                lst[j] = r

        def worker() -> None:
            # First try to acquire a fresh header; if none, attempt work stealing.
            h = acquire_header()
            while True:
                shared_mode = False
                if h is None:
                    # No fresh headers remain; steal from the heaviest queue
                    h = acquire_heaviest_header()
                    if h is None:
                        # Nothing left globally
                        break
                    shared_mode = True

                # Drain this header queue sequentially (possibly shared)
                while True:
                    with state_lock:
                        if by_header[h]:
                            snip = by_header[h].popleft()
                        else:
                            snip = None
                    if snip is None:
                        break
                    # Verify single snippet
                    results = self._verify_batch([snip])
                    if results:
                        record_result(results[0])
                        if on_progress is not None:
                            try:
                                on_progress(1)
                            except Exception:
                                pass

                # Only release if we truly owned the assignment
                if not shared_mode:
                    release_header(h)

                # Try to get another fresh header; if none, loop will steal again
                h = acquire_header()

        # Launch up to num_slots workers (bounded by number of distinct headers)
        max_workers = min(num_slots, len(headers)) if headers else 0
        if max_workers == 0:
            return {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(worker) for _ in range(max_workers)]
            for f in futs:
                f.result()

        return grouped

    def verify_simple_batch(
        self,
        items: List[Dict],
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Dict]:
        """
        Simple batch verification for complete Lean code (no wrappers needed).
        
        items: list of {id: "i:j", code: str}
        Returns: dict mapping id -> response dict (backward-compatible shape)
        
        This is useful for verifying model outputs that are already complete
        Lean files (e.g., minif2f-test outputs with cleaned_proof).
        """
        snippet_items: List[Snippet] = []
        for item in items:
            snippet_items.append(Snippet(id=str(item['id']), code=item['code']))

        # Process in batches with progress updates
        results_dict = {}
        bs = max(1, self._batch_size)
        
        for start in range(0, len(snippet_items), bs):
            chunk = snippet_items[start:start+bs]
            flat_results = self._verify_batch(chunk)
            
            # Store results by id
            for r in flat_results:
                cid = r.get("custom_id")
                results_dict[str(cid)] = r
            
            # Update progress
            if on_progress is not None:
                try:
                    on_progress(len(chunk))
                except Exception:
                    pass
        
        return results_dict
