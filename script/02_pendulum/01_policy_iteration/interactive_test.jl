using Gtk

# Initialize GTK
# Gtk.init()

# Create a new window
win = GtkWindow("Simple Gtk.jl Example", 400, 300)

# Create a button
btn = GtkButton("Click Me")

# Define a callback function for the button click event
function on_button_clicked(widget)
    println("Button was clicked!")
end

# Connect the callback function to the button click event
signal_connect(on_button_clicked, btn, "clicked")

# Add the button to the window
push!(win, btn)

# Show the window
showall(win)

# Start the GTK main event loop
Gtk.main()