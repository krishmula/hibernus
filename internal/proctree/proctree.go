// Package proctree reads the process tree under one process from /proc and writes it to a log.
//
// The Supervisor logs the tree before a checkpoint, after the process snapshot, before a restore,
// and after a restore. Reading /proc directly needs no pstree binary, and tests can point it at a
// fake /proc folder.
package proctree

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// pageKB is the size of a memory page in /proc/<pid>/stat's rss field. It is 4 KB on every
// platform Hibernus runs on.
const pageKB = 4

// Node is one process and the processes started by it.
type Node struct {
	PID, PPID, PGID, SID int
	State                string // R running, S sleeping, D disk wait, T stopped, Z zombie
	Comm                 string
	Cmdline              []string
	Threads              int
	FDs                  int
	RSSKB                int64
	Children             []*Node
}

// Entry is one row of a flattened tree.
type Entry struct {
	PID   int
	Depth int
	Node  *Node
}

// Snapshot reads the tree under rootPID from procRoot, which is /proc on a real host. It fails when
// the root does not exist. A process that exits while the walk runs is left out.
func Snapshot(procRoot string, rootPID int) (*Node, error) {
	nodes, err := readAll(procRoot)
	if err != nil {
		return nil, err
	}
	root, ok := nodes[rootPID]
	if !ok {
		return nil, fmt.Errorf("process %d does not exist", rootPID)
	}
	byParent := map[int][]*Node{}
	for _, n := range nodes {
		byParent[n.PPID] = append(byParent[n.PPID], n)
	}
	var attach func(*Node)
	attach = func(n *Node) {
		children := byParent[n.PID]
		sort.Slice(children, func(i, j int) bool { return children[i].PID < children[j].PID })
		for _, c := range children {
			if c.PID == n.PID {
				continue
			}
			n.Children = append(n.Children, c)
			attach(c)
		}
	}
	attach(root)
	return root, nil
}

func readAll(procRoot string) (map[int]*Node, error) {
	entries, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, err
	}
	nodes := map[int]*Node{}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || !e.IsDir() {
			continue
		}
		if n, err := readNode(filepath.Join(procRoot, e.Name()), pid); err == nil {
			nodes[pid] = n
		}
	}
	return nodes, nil
}

func readNode(dir string, pid int) (*Node, error) {
	stat, err := os.ReadFile(filepath.Join(dir, "stat"))
	if err != nil {
		return nil, err
	}
	n, err := parseStat(pid, string(stat))
	if err != nil {
		return nil, err
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "cmdline")); err == nil {
		n.Cmdline = splitCmdline(raw)
	}
	if fds, err := os.ReadDir(filepath.Join(dir, "fd")); err == nil {
		n.FDs = len(fds)
	}
	return n, nil
}

// parseStat reads /proc/<pid>/stat. The command name sits in parentheses and may itself hold spaces
// and parentheses, so the fields after it start after the last ")".
func parseStat(pid int, stat string) (*Node, error) {
	open := strings.Index(stat, "(")
	closing := strings.LastIndex(stat, ")")
	if open < 0 || closing < open {
		return nil, fmt.Errorf("process %d: stat has no command name", pid)
	}
	fields := strings.Fields(stat[closing+1:])
	// fields[0] is the state, which is field 3 of proc(5), so field n is fields[n-3].
	const rssField = 24
	if len(fields) < rssField-2 {
		return nil, fmt.Errorf("process %d: stat has %d fields after the name, want at least %d", pid, len(fields), rssField-2)
	}
	num := func(field int) int {
		v, _ := strconv.Atoi(fields[field-3])
		return v
	}
	rss, _ := strconv.ParseInt(fields[rssField-3], 10, 64)
	return &Node{
		PID: pid, PPID: num(4), PGID: num(5), SID: num(6),
		State:   fields[0],
		Comm:    stat[open+1 : closing],
		Threads: num(20),
		RSSKB:   rss * pageKB,
	}, nil
}

func splitCmdline(raw []byte) []string {
	text := strings.TrimRight(string(raw), "\x00")
	if text == "" {
		return nil
	}
	return strings.Split(text, "\x00")
}

// Flatten lists the tree depth first, parents before children.
func (n *Node) Flatten() []Entry {
	var out []Entry
	var walk func(*Node, int)
	walk = func(node *Node, depth int) {
		out = append(out, Entry{PID: node.PID, Depth: depth, Node: node})
		for _, c := range node.Children {
			walk(c, depth+1)
		}
	}
	walk(n, 0)
	return out
}

// Render prints one line per process, indented by depth.
func (n *Node) Render() string {
	var b strings.Builder
	for _, e := range n.Flatten() {
		p := e.Node
		fmt.Fprintf(&b, "%s%s pid=%d ppid=%d pgid=%d sid=%d state=%s threads=%d fds=%d rss=%dKB  %s\n",
			strings.Repeat("  ", e.Depth), p.Comm, p.PID, p.PPID, p.PGID, p.SID,
			p.State, p.Threads, p.FDs, p.RSSKB, strings.Join(p.Cmdline, " "))
	}
	return b.String()
}

// Shape is the part of a tree that a restore must keep: the PIDs, command names, and who is under
// whom inside the tree. The root's own parent is left out, because a restored process has a new one.
func (n *Node) Shape() string {
	var b strings.Builder
	for _, e := range n.Flatten() {
		fmt.Fprintf(&b, "%s%d %s\n", strings.Repeat("  ", e.Depth), e.PID, e.Node.Comm)
	}
	return b.String()
}

// Append adds one labelled section to the log at path, and makes the folder when needed. It writes
// the tree, or the error that stopped the tree from being read. An empty path means no logging.
func Append(path, label string, tree *Node, readErr error) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	fmt.Fprintf(file, "=== %s  %s\n", label, time.Now().UTC().Format(time.RFC3339Nano))
	switch {
	case readErr != nil:
		fmt.Fprintf(file, "(no tree: %v)\n", readErr)
	case tree == nil:
		fmt.Fprint(file, "(no tree)\n")
	default:
		fmt.Fprint(file, tree.Render())
	}
	_, err = fmt.Fprintln(file)
	return err
}
