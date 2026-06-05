//! Interactive REPL and one-shot entry points for the CLI.

use anyhow::Context;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::agent::Agent;

/// Run a single query and print the answer.
pub async fn run_once(agent: &Agent, query: String) -> anyhow::Result<()> {
    let mut history = agent.new_history();
    let answer = agent.ask(&mut history, query).await?;
    println!("\n{answer}");
    Ok(())
}

/// Run the interactive read-eval-print loop.
pub async fn run_repl(agent: &Agent) -> anyhow::Result<()> {
    let mut editor = DefaultEditor::new().context("initializing line editor")?;
    let mut history = agent.new_history();

    println!("OIDA assistant. Ask about documents, emails, and their connections.");
    println!("Commands: /reset to clear context, /exit or Ctrl-D to quit.\n");

    loop {
        match editor.readline("oida> ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(input);

                match input {
                    "/exit" | "/quit" => break,
                    "/reset" => {
                        history = agent.new_history();
                        println!("(context cleared)\n");
                        continue;
                    }
                    _ => {}
                }

                match agent.ask(&mut history, input.to_string()).await {
                    Ok(answer) => println!("\n{answer}\n"),
                    Err(e) => eprintln!("error: {e:#}\n"),
                }
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e).context("reading input"),
        }
    }

    println!("bye");
    Ok(())
}

/// Print a short banner describing the connected tools.
pub fn print_tools(tools: &[rmcp::model::Tool]) {
    eprintln!("Connected to OIDA MCP server. Tools available:");
    for t in tools {
        let desc = t
            .description
            .as_ref()
            .map(|d| d.split('.').next().unwrap_or(d).to_string())
            .unwrap_or_default();
        eprintln!("  - {}: {desc}", t.name);
    }
    eprintln!();
}
