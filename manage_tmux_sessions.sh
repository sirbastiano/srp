#!/bin/bash
# Helper script for managing tmux training sessions
# Usage: ./manage_tmux_sessions.sh [command] [options]

set -e

SCRIPT_NAME=$(basename "$0")

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $SCRIPT_NAME [command] [options]"
    echo ""
    echo "Commands:"
    echo "  list [prefix]         List all training sessions (optionally filter by prefix)"
    echo "  attach <session>      Attach to a specific session"
    echo "  kill <session>        Kill a specific session"
    echo "  kill-all [prefix]     Kill all sessions (optionally filter by prefix)"
    echo "  logs <session>        Show session output (last 50 lines)"
    echo "  status [prefix]       Show status of all sessions"
    echo ""
    echo "Examples:"
    echo "  $SCRIPT_NAME list                    # List all tmux sessions"
    echo "  $SCRIPT_NAME list sar_train          # List sessions starting with 'sar_train'"
    echo "  $SCRIPT_NAME attach sar_train_0000   # Attach to session 'sar_train_0000'"
    echo "  $SCRIPT_NAME kill sar_train_0003     # Kill session 'sar_train_0003'"
    echo "  $SCRIPT_NAME kill-all sar_train      # Kill all 'sar_train' sessions"
    echo "  $SCRIPT_NAME status                  # Show status of all sessions"
    echo ""
}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}❌ Error: tmux is not installed${NC}"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Parse command
COMMAND=${1:-"list"}
shift || true

case "$COMMAND" in
    list)
        PREFIX=${1:-""}
        echo -e "${BLUE}📋 Tmux Sessions:${NC}"
        echo ""
        if [ -z "$PREFIX" ]; then
            tmux ls 2>/dev/null || echo "No tmux sessions found"
        else
            tmux ls 2>/dev/null | grep "$PREFIX" || echo "No sessions found with prefix: $PREFIX"
        fi
        ;;
    
    attach)
        SESSION=$1
        if [ -z "$SESSION" ]; then
            echo -e "${RED}❌ Error: Session name required${NC}"
            echo "Usage: $SCRIPT_NAME attach <session_name>"
            exit 1
        fi
        
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo -e "${GREEN}✅ Attaching to session: $SESSION${NC}"
            echo -e "${YELLOW}   Press Ctrl+B then D to detach${NC}"
            sleep 1
            tmux attach -t "$SESSION"
        else
            echo -e "${RED}❌ Session not found: $SESSION${NC}"
            echo ""
            echo "Available sessions:"
            tmux ls 2>/dev/null || echo "  No sessions found"
        fi
        ;;
    
    kill)
        SESSION=$1
        if [ -z "$SESSION" ]; then
            echo -e "${RED}❌ Error: Session name required${NC}"
            echo "Usage: $SCRIPT_NAME kill <session_name>"
            exit 1
        fi
        
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo -e "${YELLOW}⚠️  Killing session: $SESSION${NC}"
            tmux kill-session -t "$SESSION"
            echo -e "${GREEN}✅ Session killed${NC}"
        else
            echo -e "${RED}❌ Session not found: $SESSION${NC}"
        fi
        ;;
    
    kill-all)
        PREFIX=${1:-""}
        
        if [ -z "$PREFIX" ]; then
            echo -e "${RED}⚠️  Warning: This will kill ALL tmux sessions!${NC}"
            read -p "Are you sure? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled"
                exit 0
            fi
            echo -e "${YELLOW}Killing all tmux sessions...${NC}"
            tmux kill-server
            echo -e "${GREEN}✅ All sessions killed${NC}"
        else
            SESSIONS=$(tmux ls 2>/dev/null | grep "$PREFIX" | cut -d: -f1 || true)
            
            if [ -z "$SESSIONS" ]; then
                echo -e "${YELLOW}No sessions found with prefix: $PREFIX${NC}"
                exit 0
            fi
            
            echo -e "${YELLOW}⚠️  The following sessions will be killed:${NC}"
            echo "$SESSIONS"
            echo ""
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled"
                exit 0
            fi
            
            COUNT=0
            for session in $SESSIONS; do
                echo "Killing: $session"
                tmux kill-session -t "$session" 2>/dev/null || true
                COUNT=$((COUNT + 1))
            done
            
            echo -e "${GREEN}✅ Killed $COUNT sessions${NC}"
        fi
        ;;
    
    logs)
        SESSION=$1
        if [ -z "$SESSION" ]; then
            echo -e "${RED}❌ Error: Session name required${NC}"
            echo "Usage: $SCRIPT_NAME logs <session_name>"
            exit 1
        fi
        
        if ! tmux has-session -t "$SESSION" 2>/dev/null; then
            echo -e "${RED}❌ Session not found: $SESSION${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}📄 Last 50 lines from session: $SESSION${NC}"
        echo "=================================================="
        tmux capture-pane -t "$SESSION" -p -S -50 || echo "Unable to capture pane"
        echo "=================================================="
        ;;
    
    status)
        PREFIX=${1:-""}
        echo -e "${BLUE}📊 Session Status:${NC}"
        echo ""
        
        if [ -z "$PREFIX" ]; then
            SESSIONS=$(tmux ls 2>/dev/null | cut -d: -f1 || true)
        else
            SESSIONS=$(tmux ls 2>/dev/null | grep "$PREFIX" | cut -d: -f1 || true)
        fi
        
        if [ -z "$SESSIONS" ]; then
            echo "No sessions found"
            exit 0
        fi
        
        printf "%-25s %-15s %-20s\n" "SESSION" "WINDOWS" "CREATED"
        echo "--------------------------------------------------------------"
        
        for session in $SESSIONS; do
            INFO=$(tmux ls 2>/dev/null | grep "^$session:" || true)
            WINDOWS=$(echo "$INFO" | sed 's/.*\[\([0-9]*\)x[0-9]*\].*/\1/' || echo "N/A")
            CREATED=$(echo "$INFO" | sed 's/.* (created \(.*\))/\1/' | cut -d')' -f1 || echo "N/A")
            printf "%-25s %-15s %-20s\n" "$session" "$WINDOWS" "$CREATED"
        done
        
        echo ""
        echo -e "${GREEN}Total: $(echo "$SESSIONS" | wc -l) sessions${NC}"
        ;;
    
    help|--help|-h)
        print_usage
        ;;
    
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
