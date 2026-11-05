// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! One-line stderr events, so a supervisor can tell a clean restart from a
//! failing one. Field values are formatted with `Display`.

/// Writes one event line: `kv-cache-service: <event> <key>=<value>...`.
macro_rules! log_event {
    ($event:literal $(, $key:literal = $value:expr)* $(,)?) => {{
        #[allow(unused_mut)]
        let mut line = ::std::string::String::from(::std::concat!("kv-cache-service: ", $event));
        $(
            line.push(' ');
            line.push_str($key);
            line.push('=');
            let _ = ::std::fmt::Write::write_fmt(&mut line, ::std::format_args!("{}", $value));
        )*
        ::std::eprintln!("{line}");
    }};
}
