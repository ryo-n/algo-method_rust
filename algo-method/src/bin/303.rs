use whiteread as w;
use std::io::Write;

fn run() -> w::reader::Result<()> {
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());

    let input = std::io::stdin();
    let input = input.lock();
    let mut input = w::Reader::new(input);

    let n: usize = input.line()?;
    let a: Vec<usize> = input.line()?;

    let mut dp = vec![100000; n];
    dp[0] = 0;

    for i in 1..n {
        let a = a[i];
        dp[i] = dp[i].min(dp[i - 1] + a);
        if i as isize - 2 >= 0 {
            dp[i] = dp[i].min(dp[i - 2] + 2 * a);
        }
    }

    writeln!(out, "{}", dp[n - 1])?;

    Ok(())
}

fn main() {
    run().unwrap()
}

// From https://github.com/krdln/whiteread on MIT license
#[allow(dead_code)]
mod whiteread {
    #![allow(bare_trait_objects)]

    use std::io;
    use std::path::Path;

    pub mod stream {
        use std::io;
        use std::str::SplitWhitespace;

        pub trait StrStream {
            fn next(&mut self) -> io::Result<Option<&str>>;
        }

        impl<'a> StrStream for SplitWhitespace<'a> {
            fn next(&mut self) -> io::Result<Option<&str>> {
                Ok(Iterator::next(self))
            }
        }

        pub struct SplitAsciiWhitespace<'a> {
            s: &'a str,
            position: usize,
        }

        impl<'a> SplitAsciiWhitespace<'a> {
            pub fn new(s: &'a str) -> Self {
                SplitAsciiWhitespace { s: s, position: 0 }
            }
            pub fn position(&self) -> usize {
                self.position
            }
            pub fn from_parts(s: &'a str, position: usize) -> Self {
                SplitAsciiWhitespace {
                    s: s,
                    position: position,
                }
            }
        }

        impl<'a> Iterator for SplitAsciiWhitespace<'a> {
            type Item = &'a str;
            fn next(&mut self) -> Option<&'a str> {
                let bytes = self.s.as_bytes();
                let mut start = self.position;
                while let Some(&c) = bytes.get(start) {
                    if c > b' ' {
                        break;
                    }
                    start += 1;
                }
                let mut end = start;
                while let Some(&c) = bytes.get(end) {
                    if c <= b' ' {
                        break;
                    }
                    end += 1;
                }
                self.position = end;
                if start != end {
                    Some(&self.s[start..end])
                } else {
                    None
                }
            }
        }

        impl<'a> StrStream for SplitAsciiWhitespace<'a> {
            fn next(&mut self) -> io::Result<Option<&str>> {
                Ok(Iterator::next(self))
            }
        }

        pub trait StrExt {
            fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace;
        }

        impl StrExt for str {
            fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace {
                SplitAsciiWhitespace::new(self)
            }
        }

        pub trait FromStream: Sized {
            fn read<I: StrStream>(it: &mut I) -> Result<Self>;
            const REQUEST_CHEAP_ERROR: bool = false;
        }

        pub type Result<T> = ::std::result::Result<T, Error>;

        #[derive(Debug)]
        pub enum Progress {
            Nothing,
            Partial,
        }

        #[derive(Debug)]
        pub enum Error {
            TooShort(Progress),
            Leftovers,
            ParseError,
            IoError(io::Error),
        }

        impl Error {
            pub fn is_too_short(&self) -> bool {
                match *self {
                    Error::TooShort(_) => true,
                    _ => false,
                }
            }
            pub fn is_nothing(&self) -> bool {
                match *self {
                    Error::TooShort(Progress::Nothing) => true,
                    _ => false,
                }
            }
            pub fn is_partial(&self) -> bool {
                match *self {
                    Error::TooShort(Progress::Partial) => true,
                    _ => false,
                }
            }
            pub fn is_leftovers(&self) -> bool {
                match *self {
                    Error::Leftovers => true,
                    _ => false,
                }
            }
            pub fn is_parse_error(&self) -> bool {
                match *self {
                    Error::ParseError => true,
                    _ => false,
                }
            }
            pub fn is_io_error(&self) -> bool {
                match *self {
                    Error::IoError(_) => true,
                    _ => false,
                }
            }
        }

        impl From<io::Error> for Error {
            fn from(e: io::Error) -> Error {
                Error::IoError(e)
            }
        }

        impl ::std::error::Error for Error {
            fn description(&self) -> &str {
                match *self {
                    Error::TooShort(Progress::Nothing) => {
                        "not enough input to start parsing a value"
                    }
                    Error::TooShort(Progress::Partial) => {
                        "not enough input to finish parsing a value"
                    }
                    Error::Leftovers => "excessive input provided",
                    Error::ParseError => "parse error occured",
                    Error::IoError(ref e) => e.description(),
                }
            }
            fn cause(&self) -> Option<&::std::error::Error> {
                #[allow(deprecated)] // Rust 1.15 doesn't have Error::source yet
                match *self {
                    Error::IoError(ref e) => e.cause(),
                    _ => None,
                }
            }
        }

        impl ::std::fmt::Display for Error {
            fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                use std::error::Error as _StdError;
                match *self {
                    Error::IoError(ref e) => e.fmt(fmt),
                    _ => fmt.write_str(self.description()),
                }
            }
        }

        impl From<Error> for io::Error {
            fn from(e: Error) -> io::Error {
                match e {
                    Error::IoError(e) => e,
                    e => io::Error::new(io::ErrorKind::InvalidData, e),
                }
            }
        }

        pub(crate) trait ResultExt {
            fn as_subsequent(self) -> Self;
        }

        impl<T> ResultExt for Result<T> {
            fn as_subsequent(mut self) -> Self {
                if let Err(Error::TooShort(ref mut kind)) = self {
                    *kind = Progress::Partial;
                }
                self
            }
        }
        macro_rules! impl_using_from_str {
            ($T:ident) => {
                impl FromStream for $T {
                    fn read<I: StrStream>(it: &mut I) -> Result<$T> {
                        it.next()?
                            .ok_or(Error::TooShort(Progress::Nothing))
                            .and_then(|s| s.parse().or(Err(Error::ParseError)))
                    }
                }
            };
        }
        impl_using_from_str!(bool);
        impl_using_from_str!(u8);
        impl_using_from_str!(u16);
        impl_using_from_str!(u32);
        impl_using_from_str!(u64);
        impl_using_from_str!(usize);
        impl_using_from_str!(i8);
        impl_using_from_str!(i16);
        impl_using_from_str!(i32);
        impl_using_from_str!(i64);
        impl_using_from_str!(isize);
        impl_using_from_str!(String);
        impl_using_from_str!(f32);
        impl_using_from_str!(f64);
        impl FromStream for char {
            fn read<I: StrStream>(it: &mut I) -> Result<char> {
                let s = it.next()?;
                s.and_then(|s| s.chars().next())
                    .ok_or(Error::TooShort(Progress::Nothing))
            }
        }

        impl FromStream for () {
            fn read<I: StrStream>(_: &mut I) -> Result<Self> {
                Ok(())
            }
        }
        macro_rules! impl_tuple {
            ( $first:ident $(, $x:ident)* ) => {
                impl< $first: FromStream $( , $x: FromStream )* > FromStream for ( $first, $( $x ),* ) {
                    fn read<I: StrStream>(it: &mut I) -> Result<Self> {
                        Ok(( $first::read(it)?, $( $x::read(it).as_subsequent()? ),* ))
                    }
                }
            };
        }
        impl_tuple!(A);
        impl_tuple!(A, B);
        impl_tuple!(A, B, C);
        impl_tuple!(A, B, C, D);
        impl_tuple!(A, B, C, D, E);
        impl_tuple!(A, B, C, D, E, F);
        impl<T: FromStream> FromStream for Option<T> {
            fn read<I: StrStream>(it: &mut I) -> Result<Option<T>> {
                match FromStream::read(it) {
                    Err(Error::TooShort(Progress::Nothing)) => Ok(None),
                    result => Ok(Some(result?)),
                }
            }
        }

        impl<T: FromStream> FromStream for Vec<T> {
            fn read<I: StrStream>(it: &mut I) -> Result<Vec<T>> {
                let mut v = vec![];
                while let Some(x) = FromStream::read(it)? {
                    v.push(x);
                }
                Ok(v)
            }
        }
    }

    pub use self::stream::FromStream;

    pub mod adapters {
        use super::stream::{FromStream, Result, ResultExt, StrStream};

        #[derive(Default, Debug, Eq, PartialEq)]
        pub struct Skip;

        impl FromStream for Skip {
            fn read<I: StrStream>(it: &mut I) -> Result<Skip> {
                it.next()?;
                Ok(Skip)
            }
        }

        #[derive(Default, Debug, Eq, PartialEq)]
        pub struct SkipAll;

        impl FromStream for SkipAll {
            fn read<I: StrStream>(it: &mut I) -> Result<SkipAll> {
                while let Some(_) = it.next()? {}
                Ok(SkipAll)
            }
        }

        #[derive(Default, Debug, Eq, PartialEq)]
        pub struct Lengthed<T>(pub Vec<T>);

        impl<T: FromStream> FromStream for Lengthed<T> {
            fn read<I: StrStream>(it: &mut I) -> Result<Lengthed<T>> {
                let sz = FromStream::read(it)?;
                let mut v = Vec::with_capacity(sz);
                loop {
                    if v.len() == sz {
                        return Ok(Lengthed(v));
                    }
                    v.push(FromStream::read(it).as_subsequent()?);
                }
            }
        }

        #[derive(Default, Debug)]
        pub struct Zeroed<T>(pub Vec<T>);

        impl<T: FromStream + Default + PartialEq> FromStream for Zeroed<T> {
            fn read<I: StrStream>(it: &mut I) -> Result<Zeroed<T>> {
                let mut v = vec![];
                let zero = Default::default();
                loop {
                    let result = FromStream::read(it);
                    let x = if v.is_empty() {
                        result?
                    } else {
                        result.as_subsequent()?
                    };
                    if x == zero {
                        return Ok(Zeroed(v));
                    } else {
                        v.push(x)
                    }
                }
            }
        }

        #[derive(Default, Debug)]
        pub struct WithCheapError<T>(pub T);

        impl<T: FromStream> FromStream for WithCheapError<T> {
            fn read<I: StrStream>(it: &mut I) -> Result<WithCheapError<T>> {
                T::read(it).map(WithCheapError)
            }
            const REQUEST_CHEAP_ERROR: bool = true;
        }
    }

    pub mod reader {
        use super::stream;
        use super::stream::Error::*;
        use super::stream::Progress;
        use super::stream::SplitAsciiWhitespace;
        use super::stream::StrStream;
        use super::FromStream;
        use std::error::Error as StdError;
        use std::fmt;
        use std::fs;
        use std::io;
        use std::path::Path;

        pub struct Reader<B: io::BufRead> {
            buf: B,
            row: u64,
            line: String,
            col: usize,
        }

        unsafe fn erase_lifetime<'a, 'b, T: 'a + 'b>(x: &'a mut T) -> &'b mut T {
            &mut *(x as *mut _)
        }

        impl<B: io::BufRead> Reader<B> {
            pub fn new(buf: B) -> Reader<B> {
                Reader {
                    buf: buf,
                    row: 0,
                    line: String::new(),
                    col: 0,
                }
            }
        }

        impl Reader<io::Empty> {
            pub(crate) fn single_line(row: u64, line: String) -> Reader<io::Empty> {
                Reader {
                    buf: io::empty(),
                    row,
                    line,
                    col: 0,
                }
            }
        }

        impl Reader<io::BufReader<io::Stdin>> {
            pub fn from_stdin_naive() -> Reader<io::BufReader<io::Stdin>> {
                Reader::new(io::BufReader::new(io::stdin()))
            }
        }

        impl Reader<io::BufReader<fs::File>> {
            pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Reader<io::BufReader<fs::File>>> {
                let file = fs::File::open(path)?;
                Ok(Reader::new(io::BufReader::new(file)))
            }
        }

        impl<B: io::BufRead> Reader<B> {
            pub fn continue_<T: FromStream>(&mut self) -> Result<T> {
                FromStream::read(self).add_lineinfo_this_type(self)
            }
            pub fn parse<T: FromStream>(&mut self) -> Result<T> {
                FromStream::read(self).add_lineinfo_this_type(self)
            }
            pub fn p<T: FromStream>(&mut self) -> T {
                self.parse().unwrap()
            }
            pub fn finish<T: FromStream>(&mut self) -> Result<T> {
                let value = self.parse()?;
                if let Ok(Some(_)) = StrStream::next(self) {
                    Err(stream::Error::Leftovers).add_lineinfo_this_type(self)
                } else {
                    Ok(value)
                }
            }
        }

        impl<B: io::BufRead> Reader<B> {
            fn read_line(&mut self) -> io::Result<Option<()>> {
                self.row += 1;
                self.line.clear();
                self.col = 0;
                let n_bytes = self.buf.read_line(&mut self.line)?;
                if n_bytes == 0 {
                    return Ok(None);
                }
                Ok(Some(()))
            }
            fn next_within_line(&mut self) -> Option<&str> {
                let mut splitter = SplitAsciiWhitespace::from_parts(&self.line, self.col);
                let ret = Iterator::next(&mut splitter);
                self.col = splitter.position();
                ret
            }
            pub fn line<T: FromStream>(&mut self) -> Result<T> {
                if let None = self.read_line()? {
                    return Err(TooShort(Progress::Nothing)).add_lineinfo::<_, T>(self);
                };
                self.finish_line()
            }
            pub fn start_line<T: FromStream>(&mut self) -> Result<T> {
                if let None = self.read_line()? {
                    return Err(TooShort(Progress::Nothing)).add_lineinfo::<_, T>(self);
                };
                self.continue_line()
            }
            pub fn continue_line<T: FromStream>(&mut self) -> Result<T> {
                let result = {
                    let mut splitter = SplitAsciiWhitespace::from_parts(&self.line, self.col);
                    let result = FromStream::read(&mut splitter);
                    self.col = splitter.position();
                    result
                };
                result.add_lineinfo_this_type(self)
            }
            pub fn finish_line<T: FromStream>(&mut self) -> Result<T> {
                let value = unsafe { erase_lifetime(self) }.continue_line()?;
                if let Some(_) = self.next_within_line() {
                    Err(Leftovers).add_lineinfo_this_type(self)
                } else {
                    Ok(value)
                }
            }
        }

        impl<B: io::BufRead> Reader<B> {
            pub fn next_line(&mut self) -> Result<&str> {
                if let None = self.read_line()? {
                    return Err(TooShort(Progress::Nothing)).add_lineinfo::<_, ()>(self);
                }
                Ok(&self.line)
            }
            pub fn into_inner(self) -> B {
                self.buf
            }
        }

        impl<B: io::BufRead> StrStream for Reader<B> {
            fn next(&mut self) -> io::Result<Option<&str>> {
                loop {
                    match unsafe { erase_lifetime(self) }.next_within_line() {
                        None => (),
                        some => return Ok(some),
                    }
                    if let None = self.read_line()? {
                        return Ok(None);
                    };
                }
            }
        }

        pub struct Error {
            error: stream::Error,
            row: u64,
            col: usize,
            rendered: Option<Box<str>>,
        }

        impl Error {
            pub fn into_inner(self) -> stream::Error {
                self.error
            }
            pub fn location(&self) -> Option<(u64, usize)> {
                if self.row > 0 {
                    Some((self.row, self.col))
                } else {
                    None
                }
            }
        }

        impl StdError for Error {
            fn description(&self) -> &str {
                self.error.description()
            }
            fn cause(&self) -> Option<&StdError> {
                Some(&self.error)
            }
        }

        impl AsRef<stream::Error> for Error {
            fn as_ref(&self) -> &stream::Error {
                &self.error
            }
        }

        impl From<io::Error> for Error {
            fn from(e: io::Error) -> Error {
                Error {
                    error: stream::Error::IoError(e),
                    row: 0,
                    col: 0,
                    rendered: None,
                }
            }
        }

        fn render_error_to_formatter<F: fmt::Write>(
            error: &stream::Error,
            line: &str,
            row: u64,
            mut col: usize,
            f: &mut F,
        ) -> fmt::Result {
            write!(f, "{}", error)?;
            #[allow(deprecated)] // Rust 1.15 doesn't have trim_end_matches yet
                let line = line.trim_right_matches(&['\n', '\r'][..]);
            if line.len() <= 120 {
                if col > line.len() {
                    col = line.len()
                }
                if (error.is_parse_error() || error.is_leftovers()) && col > 0 {
                    col -= 1;
                }
                writeln!(f, " at")?;
                let number = row.to_string();
                write!(f, "{} | ", number)?;
                writeln!(f, "{}", line)?;
                for _ in 0..number.len() + 3 {
                    write!(f, " ")?;
                }
                for c in line[..col].chars() {
                    if c <= b' ' as char {
                        write!(f, "{}", c)?;
                    } else {
                        write!(f, " ")?;
                    }
                }
                write!(f, "^")?;
            } else {
                write!(f, " at line {}, column {}", row, col + 1)?;
            }
            writeln!(f, "")?;
            Ok(())
        }

        fn render_error(error: &stream::Error, line: &str, row: u64, col: usize) -> Box<str> {
            let mut output = String::new();
            render_error_to_formatter(error, line, row, col, &mut output).unwrap();
            output.into_boxed_str()
        }

        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self.rendered {
                    Some(ref rendered) => f.write_str(rendered),
                    None => fmt::Debug::fmt(self, f),
                }
            }
        }

        impl fmt::Debug for Error {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let _ = f
                    .debug_struct("Error")
                    .field("error", &self.error)
                    .field("row", &self.row)
                    .field("col", &self.col);
                if let Some(ref rendered) = self.rendered {
                    writeln!(f, ", rendered:")?;
                    for _ in 0..79 {
                        write!(f, "~")?;
                    }
                    writeln!(f, "")?;
                    f.write_str(&rendered[..])?;
                    for _ in 0..79 {
                        write!(f, "~")?;
                    }
                    writeln!(f, "")?;
                }
                write!(f, "}}")?;
                Ok(())
            }
        }

        pub type Result<T> = ::std::result::Result<T, Error>;

        fn add_lineinfo<B, T: FromStream>(error: stream::Error, reader: &Reader<B>) -> Error
            where
                B: io::BufRead,
        {
            let rendered = if reader.row != 0 && T::REQUEST_CHEAP_ERROR == false {
                Some(render_error(&error, &reader.line, reader.row, reader.col))
            } else {
                None
            };
            Error {
                rendered: rendered,
                error: error,
                row: reader.row,
                col: reader.col,
            }
        }

        trait AddLineinfoExt<R> {
            fn add_lineinfo<B, T: FromStream>(self, reader: &Reader<B>) -> Result<R>
                where
                    B: io::BufRead;
            fn add_lineinfo_this_type<B>(self, reader: &Reader<B>) -> Result<R>
                where
                    B: io::BufRead,
                    R: FromStream,
                    Self: Sized,
            {
                self.add_lineinfo::<B, R>(reader)
            }
        }

        impl<R> AddLineinfoExt<R> for stream::Result<R> {
            fn add_lineinfo<B, T: FromStream>(self, reader: &Reader<B>) -> Result<R>
                where
                    B: io::BufRead,
            {
                self.map_err(|e| add_lineinfo::<B, T>(e, reader))
            }
        }
    }

    pub use self::reader::Reader;

    pub fn parse_line<T: FromStream>() -> reader::Result<T> {
        use std::sync::atomic;
        #[allow(deprecated)] // suggested AtomicUsize::new() doesn't work on 1.20
        static LINE_NUMBER: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;
        let row = 1 + LINE_NUMBER.fetch_add(1, atomic::Ordering::Relaxed);
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        Reader::single_line(row as u64, line).finish_line()
    }

    pub fn parse_string<T: FromStream>(s: &str) -> stream::Result<T> {
        let mut stream = stream::SplitAsciiWhitespace::new(s);
        let value = FromStream::read(&mut stream)?;
        if let Ok(Some(_)) = stream::StrStream::next(&mut stream) {
            Err(stream::Error::Leftovers)
        } else {
            Ok(value)
        }
    }

    pub fn parse_stdin<T: FromStream>() -> reader::Result<T> {
        let stdin = io::stdin();
        return Reader::new(stdin.lock()).finish();
    }

    pub fn parse_file<T: FromStream, P: AsRef<Path>>(path: P) -> reader::Result<T> {
        Ok(Reader::open(path)?.finish()?)
    }
}
